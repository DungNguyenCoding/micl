import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. Setup
# ============================================================

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1. Environment Parameters
# ============================================================

K = 300
r_cvge = 550

# Recommended diagnostic setting
P = 3             # dBm
sigma_z2 = -50    # dBm
gamma = -10       # dB

P_mW = 10 ** (P / 10.0)
sigma_z2_mW = 10 ** (sigma_z2 / 10.0)
gamma_lin = 10 ** (gamma / 10.0)

F = 1024
alpha = 4
M_0 = 400

lr = 0.05
batch_size = 10
local_epochs = 3

D = 582026
N = int(np.ceil(D / F))
pad_size = N * F - D

T_th = 0.5 * (r_cvge ** (-alpha))

# Final OTA update clipping.
# This prevents NaN crash while keeping hard-drop distortion.
max_update_norm = 3.0

print(f"D = {D}, F = {F}, N = {N}, pad_size = {pad_size}")
print(f"r_cvge = {r_cvge}")
print(f"P = {P} dBm")
print(f"P_mW = {P_mW:.4e}")
print(f"sigma_z2_mW = {sigma_z2_mW:.4e}")
print(f"gamma_lin = {gamma_lin:.4e}")
print(f"T_th = {T_th:.4e}")
print(f"max_update_norm = {max_update_norm}")


# ============================================================
# 2. Model Architecture
# ============================================================

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


_tmp_model = CNNModel()
actual_D = count_model_params(_tmp_model)
print(f"Actual model parameter count = {actual_D}")

if actual_D != D:
    raise ValueError(f"D mismatch: hard-coded D={D}, but model has {actual_D} parameters.")


# ============================================================
# 3. Dataset
# ============================================================

def get_dataset(K=300, M_0=400, mean_size=200):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)

    bs_indices = all_indices[:M_0]
    bs_dataset = Subset(train_dataset, bs_indices)

    remaining_indices = all_indices[M_0:]
    targets = train_dataset.targets.numpy()

    class_indices = {i: [] for i in range(10)}
    for idx in remaining_indices:
        class_indices[int(targets[idx])].append(int(idx))

    for c in range(10):
        np.random.shuffle(class_indices[c])

    client_datasets = []

    for k in range(K):
        N_k = max(10, np.random.poisson(mean_size))
        selected_classes = np.random.choice(10, 3, replace=False)

        samples_per_class = [N_k // 3] * 3
        for i in range(N_k % 3):
            samples_per_class[i] += 1

        client_indices = []

        for c, num_samples in zip(selected_classes, samples_per_class):
            c = int(c)

            if len(class_indices[c]) >= num_samples:
                chosen = class_indices[c][:num_samples]
                class_indices[c] = class_indices[c][num_samples:]
                client_indices.extend(chosen)

            else:
                chosen = class_indices[c]
                client_indices.extend(chosen)

                deficit = num_samples - len(chosen)
                class_indices[c] = []

                all_c_indices = [idx for idx in remaining_indices if targets[idx] == c]
                if len(all_c_indices) == 0:
                    raise RuntimeError(f"No remaining samples found for class {c}")

                extra = np.random.choice(all_c_indices, deficit, replace=True)
                client_indices.extend([int(x) for x in extra])

        np.random.shuffle(client_indices)
        client_datasets.append(Subset(train_dataset, client_indices))

    return bs_dataset, client_datasets, test_dataset


# ============================================================
# 4. Local Training Utilities
# ============================================================

def local_update(global_model, local_model, local_dataset, lr=0.05, batch_size=10, epochs=3):
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=lr)

    train_loader = DataLoader(
        local_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return {k: v.detach().clone() for k, v in local_model.state_dict().items()}


def get_flattened_delta(global_state_dict, local_state_dict):
    delta_list = []

    for key in global_state_dict.keys():
        diff = local_state_dict[key].detach() - global_state_dict[key].detach()
        delta_list.append(diff.reshape(-1))

    return torch.cat(delta_list)


def apply_flattened_update(global_model, aggregated_delta_flat):
    new_state_dict = global_model.state_dict()
    current_index = 0

    for key in new_state_dict.keys():
        shape = new_state_dict[key].shape
        num_params = new_state_dict[key].numel()

        layer_delta = aggregated_delta_flat[
            current_index: current_index + num_params
        ].view(shape)

        new_state_dict[key] = new_state_dict[key] + layer_delta.to(new_state_dict[key].device)
        current_index += num_params

    global_model.load_state_dict(new_state_dict)


def evaluate_model(model, test_dataset):
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / len(test_dataset)


def model_has_nan_or_inf(model):
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            return True, name
    return False, None


# ============================================================
# 5. Hard-Drop TCI Power Allocation
# ============================================================

def tci_power_allocation_hard_drop(u_arg, delta_arg, abs_h_sq, T_th, P_arg=None):
    """
    Hard-drop TCI.

    1. Use channel inversion only where |h|^2 >= T_th.
    2. If required power of one OFDM symbol exceeds P_arg,
       drop that whole OFDM symbol.

    This keeps harmful missing-symbol distortion.
    The final global update is clipped later to avoid NaN crash.
    """
    rho = delta_arg ** 2
    active = abs_h_sq >= T_th

    p_raw = torch.where(active, u_arg * rho, torch.zeros_like(rho))

    if P_arg is None:
        active_ratio = active.float().mean().item()
        outage_ratio = 0.0
        avg_required_power = (
            p_raw.sum(dim=1).mean().item()
            if p_raw.dim() == 2
            else p_raw.sum().item()
        )
        return p_raw, active_ratio, outage_ratio, avg_required_power

    if p_raw.dim() == 1:
        total_power = p_raw.sum()
        feasible = total_power <= P_arg

        p_out = p_raw if feasible else torch.zeros_like(p_raw)

        active_ratio = active.float().mean().item()
        outage_ratio = 0.0 if feasible else 1.0
        avg_required_power = total_power.item()

    elif p_raw.dim() == 2:
        total_power = p_raw.sum(dim=1, keepdim=True)
        feasible = total_power <= P_arg

        p_out = torch.where(feasible, p_raw, torch.zeros_like(p_raw))

        active_ratio = active.float().mean().item()
        outage_ratio = (~feasible).float().mean().item()
        avg_required_power = total_power.mean().item()

    else:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {p_raw.shape}")

    return p_out, active_ratio, outage_ratio, avg_required_power


# ============================================================
# 6. Main Training Function
# ============================================================

def run_original_bs_edge_hard_drop_tci_with_clipping(
    bs_dataset,
    client_datasets,
    test_dataset,
    K,
    D,
    F,
    P_mW=P_mW,
    r_k_dist=None,
    T_th=T_th,
    num_rounds=176,
    add_noise=True,
    max_update_norm=3.0
):
    """
    Original BS + edge OTA TCI.

    Structure:
        theta'_t = theta_t + w0 * delta_t0
        theta_{t+1} = theta'_t + estimated_edge_update

    TCI:
        hard-drop power allocation

    Safety:
        final OTA edge update is clipped before applying to the model.
    """

    global_model = CNNModel().to(device)
    reusable_local_model = CNNModel().to(device)

    Nt_list = []
    acc_list = []
    dist_list = []
    rel_dist_list = []
    active_list = []
    outage_list = []
    required_power_list = []
    power_ratio_list = []
    rho_ref_list = []
    bs_update_norm_list = []
    update_norm_list = []
    clipped_list = []

    M_0_current = len(bs_dataset)
    M_k = [len(d) for d in client_datasets]
    M_total = M_0_current + sum(M_k)

    w_0 = M_0_current / M_total
    w_k = [m / M_total for m in M_k]

    N = int(np.ceil(D / F))
    pad_size = N * F - D

    if r_k_dist is None:
        r_k_dist = np.clip(r_cvge * np.sqrt(np.random.rand(K)), 10.0, r_cvge)

    print("============================================================")
    print("Start hard-drop TCI with final update clipping")
    print(f"K = {K}")
    print(f"M0 = {M_0_current}")
    print(f"M_total = {M_total}")
    print(f"w0 = {w_0:.6e}")
    print(f"sum(w_k) = {sum(w_k):.6f}")
    print(f"N = {N}, F = {F}, Nt per round = {N}")
    print(f"P = {P} dBm")
    print(f"P_mW = {P_mW:.4e}")
    print(f"max_update_norm = {max_update_norm}")
    print("Clean BS update is applied before edge OTA aggregation.")
    print("Hard-drop TCI is used for power allocation.")
    print("Final OTA edge update is clipped before model update.")
    print("============================================================")

    diverged = False
    diverged_round = None
    diverged_reason = ""

    for t in range(num_rounds):

        # ----------------------------------------------------
        # 1. BS local update
        # ----------------------------------------------------
        theta_t0_dict = local_update(
            global_model,
            reusable_local_model,
            bs_dataset,
            lr=lr,
            batch_size=batch_size,
            epochs=local_epochs
        )

        delta_t0 = get_flattened_delta(
            global_model.state_dict(),
            theta_t0_dict
        )

        if not torch.isfinite(delta_t0).all():
            diverged = True
            diverged_round = t + 1
            diverged_reason = "delta_t0 contains NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        # theta'_t = theta_t + w0 * delta_t0
        bs_update = w_0 * delta_t0
        apply_flattened_update(global_model, bs_update)

        bs_update_norm = torch.sum(bs_update ** 2).item()

        bad_param, bad_name = model_has_nan_or_inf(global_model)
        if bad_param:
            diverged = True
            diverged_round = t + 1
            diverged_reason = f"after BS update, parameter {bad_name} contains NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        rho_ref_t = (torch.norm(delta_t0) ** 2) / D
        rho_ref_t = rho_ref_t + 1e-30

        if not torch.isfinite(rho_ref_t):
            diverged = True
            diverged_round = t + 1
            diverged_reason = "rho_ref_t is NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        # ----------------------------------------------------
        # 2. Edge OTA aggregation
        # ----------------------------------------------------
        y_t = torch.zeros((N, F), dtype=torch.complex64, device=device)
        ideal_aggregated_delta = torch.zeros(D, dtype=torch.float32, device=device)

        round_active = []
        round_outage = []
        round_required_power = []

        for k in range(K):

            theta_tk_dict = local_update(
                global_model,
                reusable_local_model,
                client_datasets[k],
                lr=lr,
                batch_size=batch_size,
                epochs=local_epochs
            )

            delta_tk = get_flattened_delta(
                global_model.state_dict(),
                theta_tk_dict
            )

            if not torch.isfinite(delta_tk).all():
                diverged = True
                diverged_round = t + 1
                diverged_reason = f"client {k} delta_tk contains NaN or Inf"
                print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
                break

            ideal_aggregated_delta += w_k[k] * delta_tk

            padded_delta_tk = torch.nn.functional.pad(
                delta_tk,
                (0, pad_size),
                mode="constant",
                value=0.0
            )

            delta_tk_N = padded_delta_tk.view(N, F)

            # Channel: h ~ CN(0, r_k^(-alpha))
            variance = float(r_k_dist[k] ** (-alpha))
            std_dev = np.sqrt(variance / 2.0)

            h_tk = torch.complex(
                torch.normal(0.0, std_dev, size=(N, F), device=device),
                torch.normal(0.0, std_dev, size=(N, F), device=device)
            )

            abs_h_sq = h_tk.real ** 2 + h_tk.imag ** 2
            g_tk = 1.0 / (abs_h_sq + 1e-30)

            u_tk = (w_k[k] ** 2) * ((gamma_lin * sigma_z2_mW) / rho_ref_t) * g_tk

            p_tk, active_ratio, outage_ratio, avg_required_power = (
                tci_power_allocation_hard_drop(
                    u_arg=u_tk,
                    delta_arg=delta_tk_N,
                    abs_h_sq=abs_h_sq,
                    T_th=T_th,
                    P_arg=P_mW
                )
            )

            if not torch.isfinite(p_tk).all():
                diverged = True
                diverged_round = t + 1
                diverged_reason = f"client {k} p_tk contains NaN or Inf"
                print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
                break

            round_active.append(active_ratio)
            round_outage.append(outage_ratio)
            round_required_power.append(avg_required_power)

            magnitude = torch.sqrt(abs_h_sq)
            phase_inversion = torch.conj(h_tk) / (magnitude + 1e-12)
            delta_sign = delta_tk_N / (torch.abs(delta_tk_N) + 1e-12)

            x_tk = (
                phase_inversion
                * delta_sign.to(torch.complex64)
                * torch.sqrt(p_tk).to(torch.complex64)
            )

            if not torch.isfinite(x_tk.real).all() or not torch.isfinite(x_tk.imag).all():
                diverged = True
                diverged_round = t + 1
                diverged_reason = f"client {k} x_tk contains NaN or Inf"
                print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
                break

            y_t += h_tk * x_tk

            del theta_tk_dict
            del delta_tk, padded_delta_tk, delta_tk_N
            del h_tk, abs_h_sq, g_tk, u_tk, p_tk, x_tk

        if diverged:
            break

        # ----------------------------------------------------
        # 3. Distortion before channel noise
        # ----------------------------------------------------
        actual_signal_at_BS = torch.sqrt(
            rho_ref_t / (gamma_lin * sigma_z2_mW)
        ) * y_t.real

        actual_signal_flat = actual_signal_at_BS.reshape(-1)[:D]

        xi_t = actual_signal_flat.detach() - ideal_aggregated_delta.detach()

        distortion_norm_sq = torch.sum(xi_t ** 2).item()
        ideal_norm_sq = torch.sum(ideal_aggregated_delta.detach() ** 2).item()
        relative_distortion = distortion_norm_sq / (ideal_norm_sq + 1e-30)

        if not np.isfinite(distortion_norm_sq) or not np.isfinite(relative_distortion):
            diverged = True
            diverged_round = t + 1
            diverged_reason = "distortion or relative distortion is NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        # ----------------------------------------------------
        # 4. Add channel noise and estimate OTA edge update
        # ----------------------------------------------------
        if add_noise:
            noise_std = np.sqrt(sigma_z2_mW / 2.0)

            z_t = torch.complex(
                torch.normal(0.0, noise_std, size=(N, F), device=device),
                torch.normal(0.0, noise_std, size=(N, F), device=device)
            )

            y_t_noisy = y_t + z_t
        else:
            y_t_noisy = y_t

        delta_sum_hat = torch.sqrt(
            rho_ref_t / (gamma_lin * sigma_z2_mW)
        ) * y_t_noisy.real

        delta_sum_flat = delta_sum_hat.reshape(-1)[:D]

        if not torch.isfinite(delta_sum_flat).all():
            diverged = True
            diverged_round = t + 1
            diverged_reason = "delta_sum_flat contains NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        # ----------------------------------------------------
        # 5. Clip final OTA edge update
        # ----------------------------------------------------
        update_norm_tensor = torch.norm(delta_sum_flat)
        update_norm = update_norm_tensor.item()
        clipped = 0

        if np.isfinite(update_norm) and update_norm > max_update_norm:
            delta_sum_flat = delta_sum_flat * (max_update_norm / (update_norm_tensor + 1e-12))
            clipped = 1

        # theta_{t+1} = theta'_t + clipped estimated edge update
        apply_flattened_update(global_model, delta_sum_flat)

        bad_param, bad_name = model_has_nan_or_inf(global_model)
        if bad_param:
            diverged = True
            diverged_round = t + 1
            diverged_reason = f"after edge update, parameter {bad_name} contains NaN or Inf"
            print(f"Divergence detected at round {diverged_round}: {diverged_reason}")
            break

        # ----------------------------------------------------
        # 6. Evaluation and logging
        # ----------------------------------------------------
        acc = evaluate_model(global_model, test_dataset)

        current_Nt = (t + 1) * N

        avg_active = float(np.mean(round_active))
        avg_outage = float(np.mean(round_outage))
        avg_required_power = float(np.mean(round_required_power))
        power_ratio = avg_required_power / P_mW

        Nt_list.append(current_Nt)
        acc_list.append(acc)
        dist_list.append(distortion_norm_sq)
        rel_dist_list.append(relative_distortion)
        active_list.append(avg_active)
        outage_list.append(avg_outage)
        required_power_list.append(avg_required_power)
        power_ratio_list.append(power_ratio)
        rho_ref_list.append(rho_ref_t.item())
        bs_update_norm_list.append(bs_update_norm)
        update_norm_list.append(update_norm)
        clipped_list.append(clipped)

        print(
            f"Round {t + 1:03d}/{num_rounds} | "
            f"Nt={current_Nt:06d} | "
            f"Acc={acc * 100:6.2f}% | "
            f"Dist={distortion_norm_sq:.4e} | "
            f"RelDist={relative_distortion:.4e} | "
            f"Active={avg_active:.4f} | "
            f"Outage={avg_outage:.4f} | "
            f"ReqP={avg_required_power:.4e} | "
            f"ReqP/P={power_ratio:.4e} | "
            f"UpdateNorm={update_norm:.4e} | "
            f"Clip={clipped} | "
            f"BSUpdateNorm={bs_update_norm:.4e} | "
            f"rho_ref={rho_ref_t.item():.4e}"
        )

        del theta_t0_dict, delta_t0, bs_update
        del y_t, ideal_aggregated_delta
        del actual_signal_at_BS, actual_signal_flat, xi_t
        del delta_sum_hat, delta_sum_flat

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if diverged:
        print("============================================================")
        print(f"Stopped early due to divergence at round {diverged_round}.")
        print(f"Reason: {diverged_reason}")
        print("============================================================")
    else:
        print("============================================================")
        print("Completed all rounds without NaN/Inf divergence.")
        print("============================================================")

    return {
        "model": global_model,
        "Nt": Nt_list,
        "acc": acc_list,
        "dist": dist_list,
        "rel_dist": rel_dist_list,
        "active": active_list,
        "outage": outage_list,
        "required_power": required_power_list,
        "power_ratio": power_ratio_list,
        "rho_ref": rho_ref_list,
        "bs_update_norm": bs_update_norm_list,
        "update_norm": update_norm_list,
        "clipped": clipped_list,
        "diverged": diverged,
        "diverged_round": diverged_round,
        "diverged_reason": diverged_reason,
    }


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    bs_dataset, client_datasets, test_dataset = get_dataset(
        K=K,
        M_0=M_0,
        mean_size=200
    )

    num_rounds = 176

    results = run_original_bs_edge_hard_drop_tci_with_clipping(
        bs_dataset=bs_dataset,
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        K=K,
        D=D,
        F=F,
        P_mW=P_mW,
        r_k_dist=None,
        T_th=T_th,
        num_rounds=num_rounds,
        add_noise=True,
        max_update_norm=max_update_norm
    )

    Nt = np.array(results["Nt"])
    acc = np.array(results["acc"])
    dist = np.array(results["dist"])
    rel_dist = np.array(results["rel_dist"])
    active = np.array(results["active"])
    outage = np.array(results["outage"])
    required_power = np.array(results["required_power"])
    power_ratio = np.array(results["power_ratio"])
    rho_ref = np.array(results["rho_ref"])
    bs_update_norm = np.array(results["bs_update_norm"])
    update_norm = np.array(results["update_norm"])
    clipped = np.array(results["clipped"])

    min_len = min(
        len(Nt),
        len(acc),
        len(dist),
        len(rel_dist),
        len(active),
        len(outage),
        len(required_power),
        len(power_ratio),
        len(rho_ref),
        len(bs_update_norm),
        len(update_norm),
        len(clipped)
    )

    Nt = Nt[:min_len]
    acc = acc[:min_len]
    dist = dist[:min_len]
    rel_dist = rel_dist[:min_len]
    active = active[:min_len]
    outage = outage[:min_len]
    required_power = required_power[:min_len]
    power_ratio = power_ratio[:min_len]
    rho_ref = rho_ref[:min_len]
    bs_update_norm = bs_update_norm[:min_len]
    update_norm = update_norm[:min_len]
    clipped = clipped[:min_len]

    np.savez(
        "hard_drop_tci_with_update_clipping_results.npz",
        Nt=Nt,
        acc=acc,
        dist=dist,
        rel_dist=rel_dist,
        active=active,
        outage=outage,
        required_power=required_power,
        power_ratio=power_ratio,
        rho_ref=rho_ref,
        bs_update_norm=bs_update_norm,
        update_norm=update_norm,
        clipped=clipped,
        diverged=np.array([results["diverged"]]),
        diverged_round=np.array([
            -1 if results["diverged_round"] is None else results["diverged_round"]
        ]),
    )

    if len(Nt) > 0:
        plt.figure()
        plt.plot(Nt, acc, label=f"Hard-drop TCI + update clipping, P={P} dBm")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Accuracy for test dataset")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_accuracy.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, dist, label="Aggregated distortion")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel(r"$\|\xi_t\|^2$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_distortion.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, rel_dist, label="Relative distortion")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Relative aggregated distortion")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_relative_distortion.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(Nt, outage, label="OFDM symbol outage ratio")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Outage ratio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_outage.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, power_ratio, label="ReqP / P")
        plt.axhline(1.0, linestyle="--", label="Power budget boundary")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Required power / available power")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_power_ratio.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, update_norm, label="OTA update norm before clipping")
        plt.axhline(max_update_norm, linestyle="--", label="Clipping threshold")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Update norm")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_update_norm.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(Nt, clipped, label="Update clipped flag")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("Clipped")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_flag.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, rho_ref, label=r"$\rho_{\mathrm{ref},t}$")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel(r"$\rho_{\mathrm{ref},t}$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_rho_ref.png", dpi=300)
        plt.close()

        plt.figure()
        plt.semilogy(Nt, bs_update_norm, label=r"$\|w_0 \Delta_{t,0}\|^2$")
        plt.xlabel("Number of symbol transmissions, Nt")
        plt.ylabel("BS update norm")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hard_drop_tci_clipped_bs_update_norm.png", dpi=300)
        plt.close()

    print("Saved:")
    print("  hard_drop_tci_with_update_clipping_results.npz")
    print("  hard_drop_tci_clipped_accuracy.png")
    print("  hard_drop_tci_clipped_distortion.png")
    print("  hard_drop_tci_clipped_relative_distortion.png")
    print("  hard_drop_tci_clipped_outage.png")
    print("  hard_drop_tci_clipped_power_ratio.png")
    print("  hard_drop_tci_clipped_update_norm.png")
    print("  hard_drop_tci_clipped_flag.png")
    print("  hard_drop_tci_clipped_rho_ref.png")
    print("  hard_drop_tci_clipped_bs_update_norm.png")

    if results["diverged"]:
        print(
            f"Run diverged at round {results['diverged_round']}: "
            f"{results['diverged_reason']}"
        )