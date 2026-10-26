#!/usr/bin/env python3
"""BayesFL cleanup PREVIEW ONLY (no move, rename, unlink, or rmtree).

Snapshot: 2026-10-26, based on the user's 163-row classification.
The embedded manifest makes this script independent of downloaded companion CSVs.

Usage:
  cd ~/DungNDH/micl/bayesfl
  python scripts/bayesfl_cleanup_dry_run.py

Read-only scanning of project files. ONLY creates .txt/.csv reports OUTSIDE bayesfl.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import os
from pathlib import Path

# Exact paths captured in the October 26 inventory.
# COLUMNS: category, snapshot bytes, file count, metrics round, resume round, path.
# Archive eligibility is STRICT: path, size, file count, and saved rounds must agree.
SNAPSHOT_TSV = r"""
ARCHIVE_CIFAR_ABLATION	219718742	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_225052
ARCHIVE_CIFAR_ABLATION	219582066	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_225051
ARCHIVE_CIFAR_ABLATION	219955769	329	281	281	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260927_193202
ARCHIVE_CIFAR_ABLATION	206633739	314	268	268	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_131743
ARCHIVE_CIFAR_ABLATION	185629622	280	237	237	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_142348
ARCHIVE_CIFAR_ABLATION	157196418	236	197	197	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_000657
ARCHIVE_CIFAR_ABLATION	135993353	205	169	169	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_local_global_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_232954
ARCHIVE_CIFAR_ABLATION	221803895	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260929_053138
ARCHIVE_CIFAR_HALF_DATA	122525472	356	300	300	outputs/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f0p5_seed0_e10_b32_lr001_budget150densefola_20260927_103154
ARCHIVE_CIFAR_HALF_DATA	52642762	174	150	-	outputs/run_cifar_basiccnn_a0p01_official_fedavg_n20_f0p5_seed0_e10_b32_lr001_r150_20260923_075942
ARCHIVE_CIFAR_HALF_DATA	100871305	175	150	-	outputs/run_cifar_basiccnn_a0p01_official_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p01_r150_20260923_092834
ARCHIVE_CIFAR_HALF_DATA	52683837	174	150	-	outputs/run_cifar_basiccnn_a0p1_official_fedavg_n20_f0p5_seed0_e10_b32_lr002_r150_20260923_063306
ARCHIVE_CIFAR_HALF_DATA	101639607	175	150	-	outputs/run_cifar_basiccnn_a0p1_official_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p1_r150_20260923_080058
ARCHIVE_CIFAR_HALF_DATA	221181503	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260929_053208
ARCHIVE_CIFAR_HALF_DATA	226126297	337	281	281	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_022548
ARCHIVE_CIFAR_HALF_DATA	560419937	865	760	760	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_073352
ARCHIVE_CIFAR_HALF_DATA	308332029	468	399	399	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_081355
ARCHIVE_CIFAR_HALF_DATA	189184363	274	222	222	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_011007
ARCHIVE_CIFAR_HALF_DATA	139529140	213	169	169	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_013239
ARCHIVE_CIFAR_HALF_DATA	220774223	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260929_090409
ARCHIVE_CIFAR_HALF_DATA	221458022	329	281	281	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_055248
ARCHIVE_CIFAR_HALF_DATA	207600902	314	268	268	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_104623
ARCHIVE_CIFAR_HALF_DATA	186536708	280	237	237	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_110454
ARCHIVE_CIFAR_HALF_DATA	158457440	236	197	197	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_033234
ARCHIVE_CIFAR_HALF_DATA	136640750	205	169	169	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_kl_local_global_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_033434
ARCHIVE_CIFAR_HALF_DATA	217797432	337	289	289	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p02_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260929_123528
ARCHIVE_CIFAR_HALF_DATA	219169905	337	281	281	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_092135
ARCHIVE_CIFAR_HALF_DATA	491779098	865	760	760	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_140023
ARCHIVE_CIFAR_HALF_DATA	282187570	468	399	399	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_135522
ARCHIVE_CIFAR_HALF_DATA	181709555	274	222	222	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_055330
ARCHIVE_CIFAR_HALF_DATA	135934703	213	169	169	outputs/run_sparse_cifar_basiccnn_a0p01_f0p5_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260926_053358
ARCHIVE_CIFAR_OLDER_BASELINES	52664203	174	150	-	outputs/run_cifar_basiccnn_a0p01_official_fedavg_n20_f1_seed0_e10_b32_lr001_r150_20260922_220548
ARCHIVE_CIFAR_OLDER_BASELINES	101553315	175	150	-	outputs/run_cifar_basiccnn_a0p01_official_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150_20260923_044012
ARCHIVE_CIFAR_OLDER_BASELINES	52715327	174	150	-	outputs/run_cifar_basiccnn_a0p1_official_fedavg_n20_f1_seed0_e10_b32_lr002_r150_20260922_155235
ARCHIVE_CIFAR_OLDER_BASELINES	101180799	175	150	-	outputs/run_cifar_basiccnn_a0p1_official_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p1_r150_20260923_030536
ARCHIVE_MNIST_OLDER_BASELINES	135691409	356	300	300	outputs/run_budgetmatch_mnist_fixed1c_s10_fedavg_n100_seed0_e10_b32_lr001_budget150densefola_20260927_133154
ARCHIVE_MNIST_OLDER_BASELINES	34298475	174	150	-	outputs/run_mnist_a0p01_official_fedavg_n100_s10_f0p016667_seed0_e10_b32_lr001_r150_20260922_191028
ARCHIVE_MNIST_OLDER_BASELINES	34364311	174	150	-	outputs/run_mnist_a0p01_official_fedavg_n100_s75_f0p125_seed0_e10_b32_lr001_r150_20260922_155235
ARCHIVE_MNIST_OLDER_BASELINES	55059652	175	150	-	outputs/run_mnist_a0p01_official_fola_n100_s10_f0p016667_seed0_e10_b32_lr001_lam0p01_r150_20260922_202723
ARCHIVE_MNIST_OLDER_BASELINES	53781278	175	150	-	outputs/run_mnist_a0p01_official_fola_n100_s75_f0p125_seed0_e10_b32_lr001_lam0p01_r150_20260922_171159
ARCHIVE_MNIST_OLDER_BASELINES	34239852	174	150	-	outputs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150_20260923_131518
ARCHIVE_MNIST_OLDER_BASELINES	55578462	175	150	-	outputs/run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150_20260923_131518
ARCHIVE_MNIST_OLDER_BASELINES	76966788	239	200	200	outputs/run_mnist_stage17_oldmlp_fixed1c_s5_n20_fola_seed0_e5_b128_lr0p01_lam0p01_r200_20261009_123003
ARCHIVE_MNIST_OLDER_BASELINES	66220115	282	240	240	outputs/run_multicast_mnist_fixed1c_s10_fedavg_mlp256x5_n100_seed0_e10_b32_lr001_r240_budget120densefola_20260930_014849
ARCHIVE_MNIST_OLDER_BASELINES	49417673	151	120	120	outputs/run_multicast_mnist_fixed1c_s10_fola_mlp256x5_n100_seed0_e10_b32_lr001_lam0p01_r120_20260930_014849
ARCHIVE_N100_MULTISEED	43544947	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed0_fedavg_clip_r400_20261010_065858
ARCHIVE_N100_MULTISEED	43635531	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed0_fedavg_noclip_r400_20261010_051927
ARCHIVE_N100_MULTISEED	30471782	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed0_fola_clip_r200_20261010_042926
ARCHIVE_N100_MULTISEED	30700010	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed0_fola_noclip_r200_20261010_033925
ARCHIVE_N100_MULTISEED	43543190	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed1_fedavg_clip_r400_20261010_120534
ARCHIVE_N100_MULTISEED	43644709	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed1_fedavg_noclip_r400_20261010_102532
ARCHIVE_N100_MULTISEED	30688427	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed1_fola_clip_r200_20261010_093531
ARCHIVE_N100_MULTISEED	30729326	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed1_fola_noclip_r200_20261010_084430
ARCHIVE_N100_MULTISEED	43552297	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed2_fedavg_clip_r400_20261010_115804
ARCHIVE_N100_MULTISEED	43640527	458	400	400	outputs/run_mnist_stage20_n100_cnn_s10_seed2_fedavg_noclip_r400_20261010_101832
ARCHIVE_N100_MULTISEED	30725669	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed2_fola_clip_r200_20261010_092831
ARCHIVE_N100_MULTISEED	30793066	239	200	200	outputs/run_mnist_stage20_n100_cnn_s10_seed2_fola_noclip_r200_20261010_083800
ARCHIVE_N100_MULTISEED	43490902	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed0_fedavg_clip_r400_20261009_210018
ARCHIVE_N100_MULTISEED	43591127	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed0_fedavg_noclip_r400_20261009_191916
ARCHIVE_N100_MULTISEED	30451907	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed0_fola_clip_r200_20261009_182915
ARCHIVE_N100_MULTISEED	30751687	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed0_fola_noclip_r200_20261009_173844
ARCHIVE_N100_MULTISEED	43486129	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed1_fedavg_clip_r400_20261009_205847
ARCHIVE_N100_MULTISEED	43583011	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed1_fedavg_noclip_r400_20261009_191816
ARCHIVE_N100_MULTISEED	30653619	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed1_fola_clip_r200_20261009_182845
ARCHIVE_N100_MULTISEED	30843375	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed1_fola_noclip_r200_20261009_173844
ARCHIVE_N100_MULTISEED	43491340	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed2_fedavg_clip_r400_20261010_020153
ARCHIVE_N100_MULTISEED	43584343	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed2_fedavg_noclip_r400_20261010_002121
ARCHIVE_N100_MULTISEED	30694386	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed2_fola_clip_r200_20261009_233120
ARCHIVE_N100_MULTISEED	30751127	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed2_fola_noclip_r200_20261009_224119
ARCHIVE_N100_MULTISEED	43494418	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed3_fedavg_clip_r400_20261010_015923
ARCHIVE_N100_MULTISEED	43588623	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed3_fedavg_noclip_r400_20261010_001851
ARCHIVE_N100_MULTISEED	30469595	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed3_fola_clip_r200_20261009_232850
ARCHIVE_N100_MULTISEED	30664746	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed3_fola_noclip_r200_20261009_223919
ARCHIVE_N100_MULTISEED	43473517	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed4_fedavg_clip_r400_20261010_070229
ARCHIVE_N100_MULTISEED	43574604	458	400	400	outputs/run_mnist_stage20_n100_cnn_s5_seed4_fedavg_noclip_r400_20261010_052226
ARCHIVE_N100_MULTISEED	30526658	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed4_fola_clip_r200_20261010_043256
ARCHIVE_N100_MULTISEED	30705180	239	200	200	outputs/run_mnist_stage20_n100_cnn_s5_seed4_fola_noclip_r200_20261010_034325
ARCHIVE_N20_DENSE_ABLATIONS	15353764	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fedavg_seed1_e5_b10_lr0p10_r300_20261008_162417
ARCHIVE_N20_DENSE_ABLATIONS	15351954	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fedavg_seed2_e5_b10_lr0p10_r300_20261008_162347
ARCHIVE_N20_DENSE_ABLATIONS	15350988	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fedavg_seed3_e5_b10_lr0p10_r300_20261008_164217
ARCHIVE_N20_DENSE_ABLATIONS	15354727	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fedavg_seed4_e5_b10_lr0p10_r300_20261008_164216
ARCHIVE_N20_DENSE_ABLATIONS	12951978	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fola_seed1_e5_b10_lr0p10_lam0p01_r150_20261008_161746
ARCHIVE_N20_DENSE_ABLATIONS	12947969	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fola_seed2_e5_b10_lr0p10_lam0p01_r150_20261008_161746
ARCHIVE_N20_DENSE_ABLATIONS	12806402	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fola_seed3_e5_b10_lr0p10_lam0p01_r150_20261008_163547
ARCHIVE_N20_DENSE_ABLATIONS	12835470	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s10_n20_fola_seed4_e5_b10_lr0p10_lam0p01_r150_20261008_163546
ARCHIVE_N20_DENSE_ABLATIONS	15332640	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed1_e5_b5_lr0p03_r300_20261008_173518
ARCHIVE_N20_DENSE_ABLATIONS	15334978	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed1_e5_b5_lr0p10_r300_20261008_165947
ARCHIVE_N20_DENSE_ABLATIONS	15331826	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed2_e5_b5_lr0p03_r300_20261008_173318
ARCHIVE_N20_DENSE_ABLATIONS	15336810	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed2_e5_b5_lr0p10_r300_20261008_165917
ARCHIVE_N20_DENSE_ABLATIONS	15330656	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed3_e5_b5_lr0p03_r300_20261008_175219
ARCHIVE_N20_DENSE_ABLATIONS	15333211	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed4_e5_b5_lr0p03_r300_20261008_175119
ARCHIVE_N20_DENSE_ABLATIONS	15334634	348	300	300	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed4_e5_b5_lr0p10_r300_20261008_171618
ARCHIVE_N20_DENSE_ABLATIONS	12751114	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed1_e5_b5_lr0p03_lam0p01_r150_20261008_172918
ARCHIVE_N20_DENSE_ABLATIONS	12913380	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed1_e5_b5_lr0p10_lam0p01_r150_20261008_165348
ARCHIVE_N20_DENSE_ABLATIONS	12866595	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed2_e5_b5_lr0p03_lam0p01_r150_20261008_172718
ARCHIVE_N20_DENSE_ABLATIONS	12918925	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed2_e5_b5_lr0p10_lam0p01_r150_20261008_165317
ARCHIVE_N20_DENSE_ABLATIONS	12669204	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed3_e5_b5_lr0p03_lam0p01_r150_20261008_174619
ARCHIVE_N20_DENSE_ABLATIONS	12737715	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed3_e5_b5_lr0p10_lam0p01_r150_20261008_171118
ARCHIVE_N20_DENSE_ABLATIONS	12620317	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed4_e5_b5_lr0p03_lam0p01_r150_20261008_174448
ARCHIVE_N20_DENSE_ABLATIONS	12789887	184	150	150	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fola_seed4_e5_b5_lr0p10_lam0p01_r150_20261008_171018
ARCHIVE_N20_DENSE_ABLATIONS	16834971	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed1_e5_b10_lr0p10_lam0p01_r200_20261009_155413
ARCHIVE_N20_DENSE_ABLATIONS	16915436	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed2_e5_b10_lr0p10_lam0p01_r200_20261009_155313
ARCHIVE_N20_DENSE_ABLATIONS	16789163	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed3_e5_b10_lr0p10_lam0p01_r200_20261009_160243
ARCHIVE_N20_DENSE_ABLATIONS	16943899	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed4_e5_b10_lr0p10_lam0p01_r200_20261009_160143
ARCHIVE_N20_DENSE_ABLATIONS	16835587	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed0_e5_b5_lr0p10_lam0p01_r200_20261009_152742
ARCHIVE_N20_DENSE_ABLATIONS	16878248	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed1_e5_b5_lr0p10_lam0p01_r200_20261009_152743
ARCHIVE_N20_DENSE_ABLATIONS	16922784	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed2_e5_b5_lr0p10_lam0p01_r200_20261009_153613
ARCHIVE_N20_DENSE_ABLATIONS	16814745	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed4_e5_b5_lr0p10_lam0p01_r200_20261009_154513
ARCHIVE_N20_DENSE_ABLATIONS	20118389	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed0_e5_b10_lr0p03_r400_20261009_114208
ARCHIVE_N20_DENSE_ABLATIONS	20113751	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed1_e5_b10_lr0p10_r400_20261009_143630
ARCHIVE_N20_DENSE_ABLATIONS	20112464	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed2_e5_b10_lr0p10_r400_20261009_143530
ARCHIVE_N20_DENSE_ABLATIONS	20111781	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed3_e5_b10_lr0p10_r400_20261009_145200
ARCHIVE_N20_DENSE_ABLATIONS	20112352	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed4_e5_b10_lr0p10_r400_20261009_145030
ARCHIVE_N20_DENSE_ABLATIONS	20096878	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed0_e5_b5_lr0p03_r400_20261009_114108
ARCHIVE_N20_DENSE_ABLATIONS	20092893	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed0_e5_b5_lr0p10_r400_20261009_115608
ARCHIVE_N20_DENSE_ABLATIONS	20090666	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed1_e5_b5_lr0p10_r400_20261009_140429
ARCHIVE_N20_DENSE_ABLATIONS	20086819	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed2_e5_b5_lr0p10_r400_20261009_140429
ARCHIVE_N20_DENSE_ABLATIONS	20090992	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed4_e5_b5_lr0p10_r400_20261009_142000
ARCHIVE_N20_DENSE_ABLATIONS	94077292	458	400	400	outputs/run_stage16_paper00_fedavg_oldmlp_s10_n20_seed0_e5_b128_lr0p01_r400_20261009_112437
ARCHIVE_N20_DENSE_ABLATIONS	94062224	458	400	400	outputs/run_stage16_paper00_fedavg_oldmlp_s5_n20_seed0_e5_b128_lr0p01_r400_20261009_112437
ARCHIVE_OLD_PLOTS	8579534	50	-	-	outputs/plots
INCOMPLETE_RESUME_SENSITIVE	23495375	185	152	152	outputs/run_mnist_stage20_n100_cnn_s10_seed3_fola_noclip_r200_20261010_134606
INCOMPLETE_RESUME_SENSITIVE	29345018	208	172	172	outputs/run_mnist_stage20_n100_cnn_s10_seed4_fola_noclip_r200_20261010_133836
KEEP_ALTERNATIVE_MNIST_S5	24740394	465	400	400	outputs/run_mnist_stage15_cnn16_32_fixed1c_s5_n20_fedavg_seed3_e5_b5_lr0p10_r300_20261008_171748
KEEP_ALTERNATIVE_MNIST_S5	16801337	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200_20261009_153613
KEEP_ALTERNATIVE_MNIST_S5	106720695	1518	1362	1362	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p05_budget150_r1362_20261010_165629
KEEP_ALTERNATIVE_MNIST_S5	76824848	1064	950	950	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p10_budget150_r950_20261010_153027
KEEP_ALTERNATIVE_MNIST_S5	42857597	567	498	498	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p25_budget150_r498_20261010_145857
KEEP_ALTERNATIVE_MNIST_S5	24472759	325	278	278	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p50_budget150_r278_20261010_144426
KEEP_ALTERNATIVE_MNIST_S5	17821112	232	193	193	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_kl_global_local_keep0p75_budget150_r193_20261010_143526
KEEP_ALTERNATIVE_MNIST_S5	109296908	1518	1362	1362	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p05_budget150_r1362_20261010_165559
KEEP_ALTERNATIVE_MNIST_S5	76540725	1064	950	950	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p10_budget150_r950_20261010_153027
KEEP_ALTERNATIVE_MNIST_S5	41175614	567	498	498	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p25_budget150_r498_20261010_145826
KEEP_ALTERNATIVE_MNIST_S5	23981410	325	278	278	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p50_budget150_r278_20261010_144426
KEEP_ALTERNATIVE_MNIST_S5	17494103	232	193	193	outputs/run_mnist_stage21_cnn_s5_n20_seed3_fola_random_keep0p75_budget150_r193_20261010_143526
KEEP_ALTERNATIVE_MNIST_S5	20086610	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s5_n20_seed3_e5_b5_lr0p10_r400_20261009_142030
KEEP_CLIPPING_ABLATION	16544091	239	200	200	outputs/run_mnist_stage24_cnn16_32_fixed1c_s10_n20_fola_clip_seed0_e5_b10_lr0p10_lam0p01_r200_20261011_094131
KEEP_MAIN_PAPER	122560313	356	300	300	outputs/run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_budget150densefola_20260927_103154
KEEP_MAIN_PAPER	116823402	184	150	150	outputs/run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150_20261010_195905
KEEP_MAIN_PAPER	16869075	239	200	200	outputs/run_mnist_stage18_cnn16_32_fola_noclip_fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200_20261009_154443
KEEP_MAIN_PAPER	105275865	1518	1362	1362	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p05_budget150_r1362_20261011_023714
KEEP_MAIN_PAPER	77295955	1064	950	950	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p10_budget150_r950_20261011_011043
KEEP_MAIN_PAPER	43537977	567	498	498	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p25_budget150_r498_20261011_003912
KEEP_MAIN_PAPER	24799240	325	278	278	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p50_budget150_r278_20261011_002442
KEEP_MAIN_PAPER	17942159	232	193	193	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_kl_global_local_keep0p75_budget150_r193_20261011_001511
KEEP_MAIN_PAPER	108678584	1518	1362	1362	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p05_budget150_r1362_20261011_023914
KEEP_MAIN_PAPER	76310027	1064	950	950	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p10_budget150_r950_20261011_011143
KEEP_MAIN_PAPER	41124484	567	498	498	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p25_budget150_r498_20261011_003942
KEEP_MAIN_PAPER	24077544	325	278	278	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p50_budget150_r278_20261011_002512
KEEP_MAIN_PAPER	17537865	232	193	193	outputs/run_mnist_stage22_cnn_s10_n20_seed0_fola_random_keep0p75_budget150_r193_20261011_001511
KEEP_MAIN_PAPER	20214151	458	400	400	outputs/run_mnist_stage23_cnn16_32_fixed1c_s10_n20_fedavg_clip_seed0_e5_b10_lr0p10_r400_20261011_091458
KEEP_MAIN_PAPER	984584640	1534	1362	1362	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260927_193202
KEEP_MAIN_PAPER	688178416	1074	950	950	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_071300
KEEP_MAIN_PAPER	375917593	578	499	499	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_090138
KEEP_MAIN_PAPER	222518517	335	278	278	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221
KEEP_MAIN_PAPER	165778744	242	193	193	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_kl_global_local_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260924_194221
KEEP_MAIN_PAPER	831930000	1535	1362	1362	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p05_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260928_024119
KEEP_MAIN_PAPER	580693041	1074	950	950	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p1_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_192526
KEEP_MAIN_PAPER	344090762	578	499	499	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p25_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_194658
KEEP_MAIN_PAPER	214909621	335	278	278	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p5_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_043532
KEEP_MAIN_PAPER	163112212	242	193	193	outputs/run_sparse_cifar_basiccnn_a0p01_f1_fola_random_keep0p75_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense_20260925_032226
KEEP_MAIN_PAPER	20112065	458	400	400	outputs/run_stage16_paper00_fedavg_cnn16_32_s10_n20_seed0_e5_b10_lr0p10_r400_20261009_115708
KEEP_SHARED_METADATA	7707251	277	-	-	outputs/partitions
""".strip()

ARCHIVE_CATEGORIES = {
    'ARCHIVE_CIFAR_ABLATION',
    'ARCHIVE_CIFAR_HALF_DATA',
    'ARCHIVE_CIFAR_OLDER_BASELINES',
    'ARCHIVE_MNIST_OLDER_BASELINES',
    'ARCHIVE_N100_MULTISEED',
    'ARCHIVE_N20_DENSE_ABLATIONS',
    'ARCHIVE_OLD_PLOTS',
}
PROTECT_CATEGORIES = {
    'KEEP_MAIN_PAPER',
    'KEEP_ALTERNATIVE_MNIST_S5',
    'KEEP_CLIPPING_ABLATION',
    'KEEP_SHARED_METADATA',
    'INCOMPLETE_RESUME_SENSITIVE',
}


def friendly(n: int) -> str:
    return f'{n / 1024**3:,.3f} GiB ({n:,} bytes)'


def snapshot_records():
    result = {}
    for line in SNAPSHOT_TSV.splitlines():
        cat, nbytes, nfiles, metric, resume, relative = line.split('\t', 5)
        assert cat in ARCHIVE_CATEGORIES | PROTECT_CATEGORIES, cat
        assert relative.startswith('outputs/')
        assert '..' not in Path(relative).parts
        assert relative not in result, relative
        result[relative] = {
            'category': cat,
            'snapshot_bytes': int(nbytes),
            'snapshot_files': int(nfiles),
            'snapshot_metric': metric,
            'snapshot_checkpoint': resume,
        }
    assert len(result) == 163, len(result)
    return result


def scan_directory(folder: Path):
    """Count regular file bytes, ignoring symlink destinations and detecting links."""
    nbytes, nfiles, links = 0, 0, 0
    for current, directory_names, file_names in os.walk(folder, followlinks=False):
        # Symlinked directories are excluded and reported as drift.
        for name in list(directory_names):
            p = Path(current) / name
            if p.is_symlink():
                directory_names.remove(name)
                links += 1
        for name in file_names:
            p = Path(current) / name
            try:
                if p.is_symlink():
                    links += 1
                elif p.is_file():
                    nfiles += 1
                    nbytes += p.stat().st_size
            except OSError:
                links += 1  # Treat unreadable entries as drift / hold.
    return nbytes, nfiles, links


def latest_metric_round(folder: Path) -> str:
    p = folder / 'metrics' / 'global_metrics.csv'
    if not p.is_file():
        return '-'
    try:
        last = None
        with p.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('round'):
                    last = int(float(row['round']))
        return str(last) if last is not None else '-'
    except (OSError, ValueError, csv.Error):
        return 'UNREADABLE'


def checkpoint_round(folder: Path) -> str:
    latest = folder / 'resume' / 'latest.json'
    if not latest.is_file():
        return '-'
    try:
        import json
        pointer = json.loads(latest.read_text(encoding='utf-8'))
        source = latest.parent / pointer['state_file']
        # Never follow state paths outside this run's resume directory.
        if source.resolve().parent != latest.parent.resolve():
            return 'UNSAFE_POINTER'
        data = json.loads(source.read_text(encoding='utf-8'))
        return str(int(data['round_id']))
    except (OSError, ValueError, KeyError, TypeError):
        return 'UNREADABLE'


def find_caches(root: Path, protected_names: set[str]):
    """Report ONLY regenerable Python cache paths outside protected outputs."""
    cache_dirs, stray_pyc = [], []
    for current, dirs, files in os.walk(root, followlinks=False):
        here = Path(current)
        rel = here.relative_to(root)
        # Never descend into experiment output folders: even their caches are protected.
        if len(rel.parts) >= 2 and rel.parts[0] == 'outputs':
            dirs.clear()
            continue
        for name in list(dirs):
            child = here / name
            if child.is_symlink():
                dirs.remove(name)
            elif name in {'__pycache__', '.pytest_cache'}:
                cache_dirs.append(child)
                dirs.remove(name)
        for name in files:
            if name.endswith('.pyc'):
                child = here / name
                if not child.is_symlink():
                    stray_pyc.append(child)
    return sorted(cache_dirs), sorted(stray_pyc)


def size_of_file(p: Path) -> int:
    try:
        return p.stat().st_size if p.is_file() and not p.is_symlink() else 0
    except OSError:
        return 0


def safe_text(s):
    return str(s).replace('\n', '\\n').replace('\r', '\\r').replace('\t', ' ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(),
                        help='Repository root; defaults to current directory')
    parser.add_argument('--report-dir', type=Path, default=None,
                        help='Must be outside the repository; defaults to repository parent')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if root.name != 'bayesfl' or not (root / 'src/bayesfl').is_dir() or \
       not (root / 'scripts/_run_gpu_queue.sh').is_file() or \
       not (root / 'outputs').is_dir():
        parser.error('Wrong directory: please cd into the bayesfl repository first.')
    dest = (args.report_dir or root.parent).expanduser().resolve()
    if dest == root or root in dest.parents:
        parser.error('Reports MUST be saved outside the bayesfl repository.')
    if not dest.exists() or not dest.is_dir():
        parser.error(f'Report directory does not exist: {dest}')

    manifest = snapshot_records()
    rows = []
    current_relatives = set()
    for child in (root / 'outputs').iterdir():
        current_relatives.add('outputs/' + child.name)

    for relative, expected in manifest.items():
        path = root / relative
        category = expected['category']
        action = 'KEEP_PROTECTED' if category in PROTECT_CATEGORIES else 'ARCHIVE_CANDIDATE'
        current_bytes, nfiles, links = 0, 0, 0
        metric, checkpoint, issues = '-', '-', []
        if path.is_symlink():
            status = 'HOLD_SYMLINK'
        elif not path.is_dir():
            status = 'MISSING'
        else:
            current_bytes, nfiles, links = scan_directory(path)
            if current_bytes != expected['snapshot_bytes']:
                issues.append('size_changed')
            if nfiles != expected['snapshot_files']:
                issues.append('file_count_changed')
            metric = latest_metric_round(path)
            checkpoint = checkpoint_round(path)
            if metric != expected['snapshot_metric']:
                issues.append('metrics_round_changed')
            if checkpoint != expected['snapshot_checkpoint']:
                issues.append('checkpoint_round_changed')
            if links:
                issues.append(f'{links}_symlinks_or_unreadable')
            status = 'MATCH' if not issues else 'HOLD_CHANGED'
        if category in PROTECT_CATEGORIES:
            action = 'KEEP_PROTECTED'  # Always protect even if status mismatches.
        elif status != 'MATCH':
            action = 'HOLD_REVIEW'      # Never recommend archive on any drift.
        rows.append(dict(
            category=category, action=action, status=status,
            relative_path=relative, snapshot_bytes=expected['snapshot_bytes'],
            current_bytes=current_bytes, snapshot_files=expected['snapshot_files'],
            current_files=nfiles, snapshot_metric_round=expected['snapshot_metric'],
            current_metric_round=metric,
            snapshot_checkpoint_round=expected['snapshot_checkpoint'],
            current_checkpoint_round=checkpoint, warning=';'.join(issues)
        ))

    # Everything added since the October 26 snapshot is protected automatically.
    for relative in sorted(current_relatives - set(manifest)):
        p = root / relative
        current_bytes, nfiles = 0, 0
        if p.is_dir() and not p.is_symlink():
            current_bytes, nfiles, links = scan_directory(p)
        elif p.is_file() and not p.is_symlink():
            current_bytes, nfiles = p.stat().st_size, 1
        rows.append(dict(
            category='NEW_UNKNOWN', action='HOLD_REVIEW', status='UNCLASSIFIED',
            relative_path=relative, snapshot_bytes=0, current_bytes=current_bytes,
            snapshot_files=0, current_files=nfiles, snapshot_metric_round='',
            current_metric_round='', snapshot_checkpoint_round='',
            current_checkpoint_round='', warning='not_in_2026_10_26_inventory'
        ))

    # Read-only cache/delete-candidate inspection. Never touch data or protected outputs.
    protected_names = {Path(p).name for p, e in manifest.items()
                       if e['category'] in PROTECT_CATEGORIES}
    cache_dirs, stray_pyc = find_caches(root, protected_names)
    delete_candidates = []
    for p in cache_dirs:
        size, nfiles, links = scan_directory(p)
        delete_candidates.append((p.relative_to(root).as_posix(), size,
                                  'REGENERABLE_CACHE' if not links else 'HOLD_SYMLINK'))
    for p in stray_pyc:
        delete_candidates.append((p.relative_to(root).as_posix(), size_of_file(p),
                                  'REGENERABLE_PYC'))

    # Archives: logs/ CONTENTS (not the directory or .gitkeep) and historical outputs/plots.
    log_files = []
    log_dir = root / 'logs'
    if log_dir.is_dir():
        for p in sorted(log_dir.iterdir()):
            if p.is_file() and not p.is_symlink() and p.name != '.gitkeep':
                log_files.append((p.relative_to(root).as_posix(), size_of_file(p)))

    optional_review = []
    cifar_tar = root / 'data/cifar-10-python.tar.gz'
    cifar_raw = root / 'data/cifar-10-batches-py'
    if cifar_tar.is_file():
        extracted_ok = all((cifar_raw / n).is_file() for n in
                           [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch'])
        optional_review.append((cifar_tar.relative_to(root).as_posix(),
                                size_of_file(cifar_tar),
                                'EXTRACTED_PRESENT_REVIEW' if extracted_ok else 'DO_NOT_DELETE'))
    mnist = root / 'data/MNIST/raw'
    if mnist.is_dir():
        for gz in sorted(mnist.glob('*.gz')):
            extracted_ok = gz.with_suffix('').is_file()
            optional_review.append((gz.relative_to(root).as_posix(), size_of_file(gz),
                                    'EXTRACTED_PRESENT_REVIEW' if extracted_ok else 'DO_NOT_DELETE'))

    archive = [r for r in rows if r['action'] == 'ARCHIVE_CANDIDATE']
    kept = [r for r in rows if r['action'] == 'KEEP_PROTECTED']
    hold = [r for r in rows if r['action'] == 'HOLD_REVIEW']
    archive_by_cat = defaultdict(lambda: [0, 0])
    for r in archive:
        v = archive_by_cat[r['category']]
        v[0] += 1
        v[1] += r['current_bytes']
    archive_bytes = sum(r['current_bytes'] for r in archive)
    logs_bytes = sum(v for _, v in log_files)
    cache_bytes = sum(v for _, v, _ in delete_candidates)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt = dest / f'bayesfl_cleanup_DRY_RUN_{stamp}.txt'
    out_csv = dest / f'bayesfl_cleanup_DRY_RUN_{stamp}.csv'
    if txt.exists() or out_csv.exists():
        parser.error('Report path collision. Try again in a second.')

    with out_csv.open('x', newline='', encoding='utf-8') as f:
        cols = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    with txt.open('x', encoding='utf-8') as f:
        w = lambda s='': f.write(str(s) + '\n')
        w('BAYESFL CLEANUP — READ-ONLY DRY RUN')
        w('=' * 110)
        w(f'Project: {root}')
        w(f'Snapshot: October 26, 2026, {len(manifest)} explicitly classified paths')
        w('NO FILES HAVE BEEN MOVED, DELETED, RENAMED, COMPRESSED OR BACKED UP.')
        w('Archive candidates require separate external backup and verification before deletion.')
        w('All newly discovered outputs, modified snapshot folders, and incomplete runs are HELD.')
        w()
        w('SUMMARY')
        w('-' * 110)
        w(f'KEEP protected folders        : {len(kept)}')
        w(f'ARCHIVE candidate folders     : {len(archive)}; {friendly(archive_bytes)}')
        w(f'HOLD / changed / new folders : {len(hold)}')
        w(f'Historical log files         : {len(log_files)}; {friendly(logs_bytes)} (archive separately)')
        w(f'Regenerable cache candidates : {len(delete_candidates)}; {friendly(cache_bytes)}')
        w(f'Optional downloaded archives: {len(optional_review)} (REVIEW ONLY)')
        w('Note: archiving on the same disk does NOT free space.')
        w('Note: archive size is POTENTIAL space reclaimed only after OFF-VOLUME backup and deletion.')
        w()
        w('A. EXACT ARCHIVE CANDIDATE FOLDERS — DO NOT DELETE YET')
        w('=' * 110)
        for cat in sorted(archive_by_cat):
            n, nbytes = archive_by_cat[cat]
            w(f'[{cat}] {n} directories; {friendly(nbytes)}')
            for r in sorted(archive, key=lambda x: x['relative_path']):
                if r['category'] == cat:
                    w(f"  {friendly(r['current_bytes']):>31}  {safe_text(r['relative_path'])}")
            w()
        w('B. PROTECTED FOLDERS (including MAIN PAPER and resumable checkpoints)')
        w('=' * 110)
        for r in sorted(kept, key=lambda r: (r['category'], r['relative_path'])):
            w(f"  {r['category']:<30} {r['status']:<14} {safe_text(r['relative_path'])}")
        w()
        w('C. HOLD — NEW, MISSING, OR CHANGED (DO NOT MOVE OR DELETE)')
        w('=' * 110)
        if not hold:
            w('  None.')
        for r in sorted(hold, key=lambda r: r['relative_path']):
            w(f"  {r['status']:<15} {safe_text(r['relative_path'])}  [{r['warning']}]")
        w()
        w('D. REGENERABLE CACHE DELETE CANDIDATES — DRY RUN ONLY')
        w('=' * 110)
        if not delete_candidates:
            w('  None found.')
        for name, nbytes, status in delete_candidates:
            w(f'  {status:<20} {friendly(nbytes):>31}  {safe_text(name)}')
        w()
        w('E. OLD LOG FILES — ARCHIVE CANDIDATES, NOT AUTO-DELETE')
        w('=' * 110)
        for name, nbytes in log_files:
            w(f'  {friendly(nbytes):>31}  {safe_text(name)}')
        w()
        w('F. OPTIONAL DATA ARCHIVES — REQUIRE DATA/BACKUP VERIFICATION')
        w('=' * 110)
        for name, nbytes, status in optional_review:
            w(f'  {status:<27} {friendly(nbytes):>31}  {safe_text(name)}')
        w()
        w('SAFETY CHECKLIST BEFORE ANY LATER CLEANUP')
        w('=' * 110)
        w('1) Confirm no experiment/queue is running; read current results.')
        w('2) Back up source/config/partitions and all archive candidates to ANOTHER disk.')
        w('3) Verify the archive by hashes or test extraction, before removing originals.')
        w('4) Preserve all KEEP and HOLD rows in their original paths.')
        w('5) Never delete outputs/partitions or all outputs/checkpoints or resume files in bulk.')
        w('6) Do not use deletion commands from this report; it contains only candidates.')
        w('PASS: REPORT-ONLY / NO PROJECT CHANGES')

    print('\n' + '=' * 75)
    print('BAYESFL CLEANUP DRY RUN — NO CHANGES TO PROJECT FILES')
    print('=' * 75)
    print(f'Protected folders : {len(kept)}')
    print(f'Archive candidates: {len(archive)} ({friendly(archive_bytes)})')
    print(f'Hold for review   : {len(hold)}')
    print(f'Old logs          : {len(log_files)} ({friendly(logs_bytes)})')
    print(f'Cache candidates  : {len(delete_candidates)} ({friendly(cache_bytes)})')
    print(f'Optional review   : {len(optional_review)}')
    print('Exact paths:', txt)
    print('CSV matrix :', out_csv)
    print('NO move/delete operations occurred.')


if __name__ == '__main__':
    main()
