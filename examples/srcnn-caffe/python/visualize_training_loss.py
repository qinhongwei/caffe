"""
This script plots the loss value.
"""

import re
import matplotlib.pyplot as plt

txt = \
"""
I0830 23:02:29.417739   796 solver.cpp:195] Iteration 20004, loss = 0.578622
I0830 23:02:29.417790   796 solver.cpp:365] Iteration 20004, lr = 1e-05
I0830 23:02:46.353974   796 solver.cpp:232] Iteration 40004, Testing net (#0)
I0830 23:02:47.138726   796 solver.cpp:270] Test score #0: 0.161416
I0830 23:02:47.142474   796 solver.cpp:287] Snapshotting to SRCNN_iter_40008
I0830 23:02:47.142845   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_40008.solverstate
I0830 23:02:47.143815   796 solver.cpp:195] Iteration 40008, loss = 0.428221
I0830 23:02:47.143838   796 solver.cpp:365] Iteration 40008, lr = 1e-05
I0830 23:03:04.043588   796 solver.cpp:232] Iteration 60006, Testing net (#0)
I0830 23:03:04.819161   796 solver.cpp:270] Test score #0: 0.133597
I0830 23:03:04.825080   796 solver.cpp:287] Snapshotting to SRCNN_iter_60012
I0830 23:03:04.825417   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_60012.solverstate
I0830 23:03:04.826032   796 solver.cpp:195] Iteration 60012, loss = 0.361884
I0830 23:03:04.826072   796 solver.cpp:365] Iteration 60012, lr = 1e-05
I0830 23:03:21.758008   796 solver.cpp:232] Iteration 80008, Testing net (#0)
I0830 23:03:22.536635   796 solver.cpp:270] Test score #0: 0.121355
I0830 23:03:22.543200   796 solver.cpp:287] Snapshotting to SRCNN_iter_80016
I0830 23:03:22.543598   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_80016.solverstate
I0830 23:03:22.544186   796 solver.cpp:195] Iteration 80016, loss = 0.328423
I0830 23:03:22.544211   796 solver.cpp:365] Iteration 80016, lr = 1e-05
I0830 23:03:39.410823   796 solver.cpp:232] Iteration 100010, Testing net (#0)
I0830 23:03:40.187535   796 solver.cpp:270] Test score #0: 0.115613
I0830 23:03:40.196040   796 solver.cpp:287] Snapshotting to SRCNN_iter_100020
I0830 23:03:40.196393   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_100020.solverstate
I0830 23:03:40.196987   796 solver.cpp:195] Iteration 100020, loss = 0.309899
I0830 23:03:40.197016   796 solver.cpp:365] Iteration 100020, lr = 1e-05
I0830 23:03:57.118890   796 solver.cpp:232] Iteration 120012, Testing net (#0)
I0830 23:03:57.898970   796 solver.cpp:270] Test score #0: 0.111475
I0830 23:03:57.909205   796 solver.cpp:287] Snapshotting to SRCNN_iter_120024
I0830 23:03:57.909582   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_120024.solverstate
I0830 23:03:57.910202   796 solver.cpp:195] Iteration 120024, loss = 0.298778
I0830 23:03:57.910228   796 solver.cpp:365] Iteration 120024, lr = 1e-05
I0830 23:04:14.830847   796 solver.cpp:232] Iteration 140014, Testing net (#0)
I0830 23:04:15.615363   796 solver.cpp:270] Test score #0: 0.107606
I0830 23:04:15.626828   796 solver.cpp:287] Snapshotting to SRCNN_iter_140028
I0830 23:04:15.627166   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_140028.solverstate
I0830 23:04:15.627758   796 solver.cpp:195] Iteration 140028, loss = 0.290971
I0830 23:04:15.627780   796 solver.cpp:365] Iteration 140028, lr = 1e-05
I0830 23:04:32.654047   796 solver.cpp:232] Iteration 160016, Testing net (#0)
I0830 23:04:33.431738   796 solver.cpp:270] Test score #0: 0.104881
I0830 23:04:33.445960   796 solver.cpp:287] Snapshotting to SRCNN_iter_160032
I0830 23:04:33.446285   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_160032.solverstate
I0830 23:04:33.446825   796 solver.cpp:195] Iteration 160032, loss = 0.285247
I0830 23:04:33.446849   796 solver.cpp:365] Iteration 160032, lr = 1e-05
I0830 23:04:50.394443   796 solver.cpp:232] Iteration 180018, Testing net (#0)
I0830 23:04:51.176180   796 solver.cpp:270] Test score #0: 0.103053
I0830 23:04:51.191655   796 solver.cpp:287] Snapshotting to SRCNN_iter_180036
I0830 23:04:51.191973   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_180036.solverstate
I0830 23:04:51.192560   796 solver.cpp:195] Iteration 180036, loss = 0.280845
I0830 23:04:51.192582   796 solver.cpp:365] Iteration 180036, lr = 1e-05
I0830 23:05:08.135113   796 solver.cpp:232] Iteration 200020, Testing net (#0)
I0830 23:05:08.926317   796 solver.cpp:270] Test score #0: 0.101469
I0830 23:05:08.944000   796 solver.cpp:287] Snapshotting to SRCNN_iter_200040
I0830 23:05:08.944427   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_200040.solverstate
I0830 23:05:08.945114   796 solver.cpp:195] Iteration 200040, loss = 0.277326
I0830 23:05:08.945152   796 solver.cpp:365] Iteration 200040, lr = 1e-05
I0830 23:05:25.926376   796 solver.cpp:232] Iteration 220022, Testing net (#0)
I0830 23:05:26.706737   796 solver.cpp:270] Test score #0: 0.100722
I0830 23:05:26.725574   796 solver.cpp:287] Snapshotting to SRCNN_iter_220044
I0830 23:05:26.725883   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_220044.solverstate
I0830 23:05:26.726430   796 solver.cpp:195] Iteration 220044, loss = 0.274449
I0830 23:05:26.726454   796 solver.cpp:365] Iteration 220044, lr = 1e-05
I0830 23:05:43.644079   796 solver.cpp:232] Iteration 240024, Testing net (#0)
I0830 23:05:44.424046   796 solver.cpp:270] Test score #0: 0.0989064
I0830 23:05:44.444973   796 solver.cpp:287] Snapshotting to SRCNN_iter_240048
I0830 23:05:44.445240   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_240048.solverstate
I0830 23:05:44.445792   796 solver.cpp:195] Iteration 240048, loss = 0.271982
I0830 23:05:44.445816   796 solver.cpp:365] Iteration 240048, lr = 1e-05
I0830 23:06:01.358248   796 solver.cpp:232] Iteration 260026, Testing net (#0)
I0830 23:06:02.135788   796 solver.cpp:270] Test score #0: 0.0977695
I0830 23:06:02.158663   796 solver.cpp:287] Snapshotting to SRCNN_iter_260052
I0830 23:06:02.158993   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_260052.solverstate
I0830 23:06:02.159603   796 solver.cpp:195] Iteration 260052, loss = 0.269841
I0830 23:06:02.159629   796 solver.cpp:365] Iteration 260052, lr = 1e-05
I0830 23:06:19.036283   796 solver.cpp:232] Iteration 280028, Testing net (#0)
I0830 23:06:19.880151   796 solver.cpp:270] Test score #0: 0.0978085
I0830 23:06:19.904516   796 solver.cpp:287] Snapshotting to SRCNN_iter_280056
I0830 23:06:19.904867   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_280056.solverstate
I0830 23:06:19.905436   796 solver.cpp:195] Iteration 280056, loss = 0.267913
I0830 23:06:19.905462   796 solver.cpp:365] Iteration 280056, lr = 1e-05
I0830 23:06:36.781340   796 solver.cpp:232] Iteration 300030, Testing net (#0)
I0830 23:06:37.560071   796 solver.cpp:270] Test score #0: 0.0977241
I0830 23:06:37.585538   796 solver.cpp:287] Snapshotting to SRCNN_iter_300060
I0830 23:06:37.585863   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_300060.solverstate
I0830 23:06:37.586901   796 solver.cpp:195] Iteration 300060, loss = 0.266174
I0830 23:06:37.586925   796 solver.cpp:365] Iteration 300060, lr = 1e-05
I0830 23:06:54.512564   796 solver.cpp:232] Iteration 320032, Testing net (#0)
I0830 23:06:55.294209   796 solver.cpp:270] Test score #0: 0.0968464
I0830 23:06:55.322038   796 solver.cpp:287] Snapshotting to SRCNN_iter_320064
I0830 23:06:55.322463   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_320064.solverstate
I0830 23:06:55.323237   796 solver.cpp:195] Iteration 320064, loss = 0.264589
I0830 23:06:55.323262   796 solver.cpp:365] Iteration 320064, lr = 1e-05
I0830 23:07:12.194263   796 solver.cpp:232] Iteration 340034, Testing net (#0)
I0830 23:07:12.991510   796 solver.cpp:270] Test score #0: 0.094625
I0830 23:07:13.021288   796 solver.cpp:287] Snapshotting to SRCNN_iter_340068
I0830 23:07:13.021567   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_340068.solverstate
I0830 23:07:13.022100   796 solver.cpp:195] Iteration 340068, loss = 0.263111
I0830 23:07:13.022124   796 solver.cpp:365] Iteration 340068, lr = 1e-05
I0830 23:07:30.003763   796 solver.cpp:232] Iteration 360036, Testing net (#0)
I0830 23:07:30.779625   796 solver.cpp:270] Test score #0: 0.0940636
I0830 23:07:30.811125   796 solver.cpp:287] Snapshotting to SRCNN_iter_360072
I0830 23:07:30.811430   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_360072.solverstate
I0830 23:07:30.811996   796 solver.cpp:195] Iteration 360072, loss = 0.261755
I0830 23:07:30.812029   796 solver.cpp:365] Iteration 360072, lr = 1e-05
I0830 23:07:47.708026   796 solver.cpp:232] Iteration 380038, Testing net (#0)
I0830 23:07:48.483667   796 solver.cpp:270] Test score #0: 0.094565
I0830 23:07:48.517200   796 solver.cpp:287] Snapshotting to SRCNN_iter_380076
I0830 23:07:48.517474   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_380076.solverstate
I0830 23:07:48.517997   796 solver.cpp:195] Iteration 380076, loss = 0.260513
I0830 23:07:48.518021   796 solver.cpp:365] Iteration 380076, lr = 1e-05
I0830 23:08:05.448905   796 solver.cpp:232] Iteration 400040, Testing net (#0)
I0830 23:08:06.236060   796 solver.cpp:270] Test score #0: 0.0936951
I0830 23:08:06.270323   796 solver.cpp:287] Snapshotting to SRCNN_iter_400080
I0830 23:08:06.270596   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_400080.solverstate
I0830 23:08:06.271121   796 solver.cpp:195] Iteration 400080, loss = 0.259338
I0830 23:08:06.271145   796 solver.cpp:365] Iteration 400080, lr = 1e-05
I0830 23:08:23.333593   796 solver.cpp:232] Iteration 420042, Testing net (#0)
I0830 23:08:24.121047   796 solver.cpp:270] Test score #0: 0.0923165
I0830 23:08:24.157490   796 solver.cpp:287] Snapshotting to SRCNN_iter_420084
I0830 23:08:24.157830   796 solver.cpp:294] Snapshotting solver state to SRCNN_iter_420084.solverstate
I0830 23:08:24.158382   796 solver.cpp:195] Iteration 420084, loss = 0.258207
I0830 23:08:24.158409   796 solver.cpp:365] Iteration 420084, lr = 1e-05
"""

new_txt = \
"""
0411 17:33:31.714489 43726 solver.cpp:189] Iteration 500100, loss = 2.59127
I0411 17:33:31.714534 43726 solver.cpp:204]     Train net output #0: loss = 2.59127 (* 1 = 2.59127 loss)
I0411 17:33:31.714545 43726 solver.cpp:464] Iteration 500100, lr = 1e-06
I0411 17:34:23.928946 43726 solver.cpp:266] Iteration 520052, Testing net (#0)
I0411 17:34:25.453881 43726 solver.cpp:315]     Test net output #0: loss = 0.591643 (* 1 = 0.591643 loss)
I0411 17:34:25.589984 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_520104.caffemodel
I0411 17:34:25.590353 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_520104.solverstate
I0411 17:34:25.591362 43726 solver.cpp:189] Iteration 520104, loss = 2.5905
I0411 17:34:25.591405 43726 solver.cpp:204]     Train net output #0: loss = 2.59049 (* 1 = 2.59049 loss)
I0411 17:34:25.591426 43726 solver.cpp:464] Iteration 520104, lr = 1e-06
I0411 17:35:17.623512 43726 solver.cpp:266] Iteration 540054, Testing net (#0)
I0411 17:35:19.162940 43726 solver.cpp:315]     Test net output #0: loss = 0.591453 (* 1 = 0.591453 loss)
I0411 17:35:19.304172 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_540108.caffemodel
I0411 17:35:19.304493 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_540108.solverstate
I0411 17:35:19.305521 43726 solver.cpp:189] Iteration 540108, loss = 2.59048
I0411 17:35:19.305554 43726 solver.cpp:204]     Train net output #0: loss = 2.59048 (* 1 = 2.59048 loss)
I0411 17:35:19.305564 43726 solver.cpp:464] Iteration 540108, lr = 1e-06
I0411 17:36:11.329741 43726 solver.cpp:266] Iteration 560056, Testing net (#0)
I0411 17:36:12.852217 43726 solver.cpp:315]     Test net output #0: loss = 0.591262 (* 1 = 0.591262 loss)
I0411 17:36:12.998541 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_560112.caffemodel
I0411 17:36:12.998889 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_560112.solverstate
I0411 17:36:12.999907 43726 solver.cpp:189] Iteration 560112, loss = 2.59054
I0411 17:36:12.999940 43726 solver.cpp:204]     Train net output #0: loss = 2.59055 (* 1 = 2.59055 loss)
I0411 17:36:12.999951 43726 solver.cpp:464] Iteration 560112, lr = 1e-06
I0411 17:37:04.882321 43726 solver.cpp:266] Iteration 580058, Testing net (#0)
I0411 17:37:06.405553 43726 solver.cpp:315]     Test net output #0: loss = 0.591099 (* 1 = 0.591099 loss)
I0411 17:37:06.557293 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_580116.caffemodel
I0411 17:37:06.557657 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_580116.solverstate
I0411 17:37:06.558692 43726 solver.cpp:189] Iteration 580116, loss = 2.59056
I0411 17:37:06.558735 43726 solver.cpp:204]     Train net output #0: loss = 2.59057 (* 1 = 2.59057 loss)
I0411 17:37:06.558747 43726 solver.cpp:464] Iteration 580116, lr = 1e-06
I0411 17:37:58.529306 43726 solver.cpp:266] Iteration 600060, Testing net (#0)
I0411 17:38:00.068238 43726 solver.cpp:315]     Test net output #0: loss = 0.590956 (* 1 = 0.590956 loss)
I0411 17:38:00.226939 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_600120.caffemodel
I0411 17:38:00.227306 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_600120.solverstate
I0411 17:38:00.228354 43726 solver.cpp:189] Iteration 600120, loss = 2.5906
I0411 17:38:00.228395 43726 solver.cpp:204]     Train net output #0: loss = 2.59061 (* 1 = 2.59061 loss)
I0411 17:38:00.228410 43726 solver.cpp:464] Iteration 600120, lr = 1e-06
I0411 17:38:52.276918 43726 solver.cpp:266] Iteration 620062, Testing net (#0)
I0411 17:38:53.792708 43726 solver.cpp:315]     Test net output #0: loss = 0.590825 (* 1 = 0.590825 loss)
I0411 17:38:53.956514 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_620124.caffemodel
I0411 17:38:53.956948 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_620124.solverstate
I0411 17:38:53.958070 43726 solver.cpp:189] Iteration 620124, loss = 2.59071
I0411 17:38:53.958120 43726 solver.cpp:204]     Train net output #0: loss = 2.5907 (* 1 = 2.5907 loss)
I0411 17:38:53.958134 43726 solver.cpp:464] Iteration 620124, lr = 1e-06
I0411 17:39:46.068542 43726 solver.cpp:266] Iteration 640064, Testing net (#0)
I0411 17:39:47.582144 43726 solver.cpp:315]     Test net output #0: loss = 0.590705 (* 1 = 0.590705 loss)
I0411 17:39:47.748901 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_640128.caffemodel
I0411 17:39:47.749245 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_640128.solverstate
I0411 17:39:47.750246 43726 solver.cpp:189] Iteration 640128, loss = 2.59063
I0411 17:39:47.750279 43726 solver.cpp:204]     Train net output #0: loss = 2.59064 (* 1 = 2.59064 loss)
I0411 17:39:47.750289 43726 solver.cpp:464] Iteration 640128, lr = 1e-06
I0411 17:40:39.666751 43726 solver.cpp:266] Iteration 660066, Testing net (#0)
I0411 17:40:41.213986 43726 solver.cpp:315]     Test net output #0: loss = 0.590578 (* 1 = 0.590578 loss)
I0411 17:40:41.387565 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_660132.caffemodel
I0411 17:40:41.387886 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_660132.solverstate
I0411 17:40:41.388883 43726 solver.cpp:189] Iteration 660132, loss = 2.5907
I0411 17:40:41.388912 43726 solver.cpp:204]     Train net output #0: loss = 2.5907 (* 1 = 2.5907 loss)
I0411 17:40:41.388923 43726 solver.cpp:464] Iteration 660132, lr = 1e-06
I0411 17:41:33.233052 43726 solver.cpp:266] Iteration 680068, Testing net (#0)
I0411 17:41:34.747557 43726 solver.cpp:315]     Test net output #0: loss = 0.59042 (* 1 = 0.59042 loss)
I0411 17:41:34.924518 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_680136.caffemodel
I0411 17:41:34.924875 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_680136.solverstate
I0411 17:41:34.925916 43726 solver.cpp:189] Iteration 680136, loss = 2.59075
I0411 17:41:34.925961 43726 solver.cpp:204]     Train net output #0: loss = 2.59077 (* 1 = 2.59077 loss)
I0411 17:41:34.925972 43726 solver.cpp:464] Iteration 680136, lr = 1e-06
I0411 17:42:26.671577 43726 solver.cpp:266] Iteration 700070, Testing net (#0)
I0411 17:42:28.188632 43726 solver.cpp:315]     Test net output #0: loss = 0.590268 (* 1 = 0.590268 loss)
I0411 17:42:28.370331 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_700140.caffemodel
I0411 17:42:28.370630 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_700140.solverstate
I0411 17:42:28.371587 43726 solver.cpp:189] Iteration 700140, loss = 2.59069
I0411 17:42:28.371618 43726 solver.cpp:204]     Train net output #0: loss = 2.59072 (* 1 = 2.59072 loss)
I0411 17:42:28.371628 43726 solver.cpp:464] Iteration 700140, lr = 1e-06
I0411 17:43:20.147362 43726 solver.cpp:266] Iteration 720072, Testing net (#0)
I0411 17:43:21.684031 43726 solver.cpp:315]     Test net output #0: loss = 0.59012 (* 1 = 0.59012 loss)
I0411 17:43:21.872601 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_720144.caffemodel
I0411 17:43:21.872936 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_720144.solverstate
I0411 17:43:21.873985 43726 solver.cpp:189] Iteration 720144, loss = 2.59065
I0411 17:43:21.874014 43726 solver.cpp:204]     Train net output #0: loss = 2.59067 (* 1 = 2.59067 loss)
I0411 17:43:21.874027 43726 solver.cpp:464] Iteration 720144, lr = 1e-06
I0411 17:44:13.866355 43726 solver.cpp:266] Iteration 740074, Testing net (#0)
I0411 17:44:15.404813 43726 solver.cpp:315]     Test net output #0: loss = 0.589979 (* 1 = 0.589979 loss)
I0411 17:44:15.598006 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_740148.caffemodel
I0411 17:44:15.598273 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_740148.solverstate
I0411 17:44:15.599217 43726 solver.cpp:189] Iteration 740148, loss = 2.59062
I0411 17:44:15.599243 43726 solver.cpp:204]     Train net output #0: loss = 2.59065 (* 1 = 2.59065 loss)
I0411 17:44:15.599253 43726 solver.cpp:464] Iteration 740148, lr = 1e-06
I0411 17:45:07.657521 43726 solver.cpp:266] Iteration 760076, Testing net (#0)
I0411 17:45:09.200563 43726 solver.cpp:315]     Test net output #0: loss = 0.589846 (* 1 = 0.589846 loss)
I0411 17:45:09.399075 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_760152.caffemodel
I0411 17:45:09.399430 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_760152.solverstate
I0411 17:45:09.400424 43726 solver.cpp:189] Iteration 760152, loss = 2.59059
I0411 17:45:09.400467 43726 solver.cpp:204]     Train net output #0: loss = 2.59062 (* 1 = 2.59062 loss)
I0411 17:45:09.400478 43726 solver.cpp:464] Iteration 760152, lr = 1e-06
I0411 17:46:01.319386 43726 solver.cpp:266] Iteration 780078, Testing net (#0)
I0411 17:46:02.838047 43726 solver.cpp:315]     Test net output #0: loss = 0.589713 (* 1 = 0.589713 loss)
I0411 17:46:03.041892 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_780156.caffemodel
I0411 17:46:03.042290 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_780156.solverstate
I0411 17:46:03.043315 43726 solver.cpp:189] Iteration 780156, loss = 2.59061
I0411 17:46:03.043349 43726 solver.cpp:204]     Train net output #0: loss = 2.59064 (* 1 = 2.59064 loss)
I0411 17:46:03.043359 43726 solver.cpp:464] Iteration 780156, lr = 1e-06
I0411 17:46:54.828497 43726 solver.cpp:266] Iteration 800080, Testing net (#0)
I0411 17:46:56.350289 43726 solver.cpp:315]     Test net output #0: loss = 0.589584 (* 1 = 0.589584 loss)
I0411 17:46:56.558931 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_800160.caffemodel
I0411 17:46:56.559278 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_800160.solverstate
I0411 17:46:56.560291 43726 solver.cpp:189] Iteration 800160, loss = 2.59071
I0411 17:46:56.560324 43726 solver.cpp:204]     Train net output #0: loss = 2.59074 (* 1 = 2.59074 loss)
I0411 17:46:56.560335 43726 solver.cpp:464] Iteration 800160, lr = 1e-06
I0411 17:47:48.320425 43726 solver.cpp:266] Iteration 820082, Testing net (#0)
I0411 17:47:49.872076 43726 solver.cpp:315]     Test net output #0: loss = 0.58949 (* 1 = 0.58949 loss)
I0411 17:47:50.085314 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_820164.caffemodel
I0411 17:47:50.085592 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_820164.solverstate
I0411 17:47:50.086562 43726 solver.cpp:189] Iteration 820164, loss = 2.59073
I0411 17:47:50.086591 43726 solver.cpp:204]     Train net output #0: loss = 2.59077 (* 1 = 2.59077 loss)
I0411 17:47:50.086601 43726 solver.cpp:464] Iteration 820164, lr = 1e-06
I0411 17:48:41.825510 43726 solver.cpp:266] Iteration 840084, Testing net (#0)
I0411 17:48:43.355486 43726 solver.cpp:315]     Test net output #0: loss = 0.589423 (* 1 = 0.589423 loss)
I0411 17:48:43.574445 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_840168.caffemodel
I0411 17:48:43.574807 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_840168.solverstate
I0411 17:48:43.575767 43726 solver.cpp:189] Iteration 840168, loss = 2.59081
I0411 17:48:43.575794 43726 solver.cpp:204]     Train net output #0: loss = 2.59086 (* 1 = 2.59086 loss)
I0411 17:48:43.575805 43726 solver.cpp:464] Iteration 840168, lr = 1e-06
I0411 17:49:35.430974 43726 solver.cpp:266] Iteration 860086, Testing net (#0)
I0411 17:49:36.979928 43726 solver.cpp:315]     Test net output #0: loss = 0.589254 (* 1 = 0.589254 loss)
I0411 17:49:37.205060 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_860172.caffemodel
I0411 17:49:37.205350 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_860172.solverstate
I0411 17:49:37.206333 43726 solver.cpp:189] Iteration 860172, loss = 2.59082
I0411 17:49:37.206360 43726 solver.cpp:204]     Train net output #0: loss = 2.59086 (* 1 = 2.59086 loss)
I0411 17:49:37.206372 43726 solver.cpp:464] Iteration 860172, lr = 1e-06
I0411 17:50:29.217483 43726 solver.cpp:266] Iteration 880088, Testing net (#0)
I0411 17:50:30.752612 43726 solver.cpp:315]     Test net output #0: loss = 0.589071 (* 1 = 0.589071 loss)
I0411 17:50:30.984365 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_880176.caffemodel
I0411 17:50:30.984733 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_880176.solverstate
I0411 17:50:30.985774 43726 solver.cpp:189] Iteration 880176, loss = 2.59075
I0411 17:50:30.985810 43726 solver.cpp:204]     Train net output #0: loss = 2.5908 (* 1 = 2.5908 loss)
I0411 17:50:30.985823 43726 solver.cpp:464] Iteration 880176, lr = 1e-06
I0411 17:51:22.809362 43726 solver.cpp:266] Iteration 900090, Testing net (#0)
I0411 17:51:24.329434 43726 solver.cpp:315]     Test net output #0: loss = 0.588921 (* 1 = 0.588921 loss)
I0411 17:51:24.562144 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_900180.caffemodel
I0411 17:51:24.562435 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_900180.solverstate
I0411 17:51:24.563395 43726 solver.cpp:189] Iteration 900180, loss = 2.59074
I0411 17:51:24.563422 43726 solver.cpp:204]     Train net output #0: loss = 2.59079 (* 1 = 2.59079 loss)
I0411 17:51:24.563432 43726 solver.cpp:464] Iteration 900180, lr = 1e-06
I0411 17:52:16.199527 43726 solver.cpp:266] Iteration 920092, Testing net (#0)
I0411 17:52:17.707990 43726 solver.cpp:315]     Test net output #0: loss = 0.588785 (* 1 = 0.588785 loss)
I0411 17:52:17.945971 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_920184.caffemodel
I0411 17:52:17.946272 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_920184.solverstate
I0411 17:52:17.947227 43726 solver.cpp:189] Iteration 920184, loss = 2.59075
I0411 17:52:17.947252 43726 solver.cpp:204]     Train net output #0: loss = 2.5908 (* 1 = 2.5908 loss)
I0411 17:52:17.947263 43726 solver.cpp:464] Iteration 920184, lr = 1e-06
I0411 17:53:09.631433 43726 solver.cpp:266] Iteration 940094, Testing net (#0)
I0411 17:53:11.147866 43726 solver.cpp:315]     Test net output #0: loss = 0.588657 (* 1 = 0.588657 loss)
I0411 17:53:11.392227 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_940188.caffemodel
I0411 17:53:11.392514 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_940188.solverstate
I0411 17:53:11.393514 43726 solver.cpp:189] Iteration 940188, loss = 2.59078
I0411 17:53:11.393540 43726 solver.cpp:204]     Train net output #0: loss = 2.59083 (* 1 = 2.59083 loss)
I0411 17:53:11.393550 43726 solver.cpp:464] Iteration 940188, lr = 1e-06
I0411 17:54:03.124546 43726 solver.cpp:266] Iteration 960096, Testing net (#0)
I0411 17:54:04.647433 43726 solver.cpp:315]     Test net output #0: loss = 0.588526 (* 1 = 0.588526 loss)
I0411 17:54:04.897806 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_960192.caffemodel
I0411 17:54:04.898107 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_960192.solverstate
I0411 17:54:04.899096 43726 solver.cpp:189] Iteration 960192, loss = 2.5909
I0411 17:54:04.899121 43726 solver.cpp:204]     Train net output #0: loss = 2.59096 (* 1 = 2.59096 loss)
I0411 17:54:04.899132 43726 solver.cpp:464] Iteration 960192, lr = 1e-06
I0411 17:54:56.837553 43726 solver.cpp:266] Iteration 980098, Testing net (#0)
I0411 17:54:58.367269 43726 solver.cpp:315]     Test net output #0: loss = 0.588395 (* 1 = 0.588395 loss)
I0411 17:54:58.624461 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_980196.caffemodel
I0411 17:54:58.625632 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_980196.solverstate
I0411 17:54:58.627074 43726 solver.cpp:189] Iteration 980196, loss = 2.591
I0411 17:54:58.627138 43726 solver.cpp:204]     Train net output #0: loss = 2.59105 (* 1 = 2.59105 loss)
I0411 17:54:58.627163 43726 solver.cpp:464] Iteration 980196, lr = 1e-06
I0411 17:55:50.426172 43726 solver.cpp:334] Snapshotting to models/srcnn/snapshots/srcnn_iter_1000001.caffemodel
I0411 17:55:50.426630 43726 solver.cpp:342] Snapshotting solver state to models/srcnn/snapshots/srcnn_iter_1000001.solverstate
I0411 17:55:50.426750 43726 solver.cpp:253] Optimization Done.
I0411 17:55:50.426767 43726 caffe.cpp:134] Optimization Done.

"""

#m = re.findall('Iteration (\d+), loss = ([\d.]+)', new_txt)
m = re.findall('Iteration (\d+)', new_txt)
test_loss = re.findall('Test net output #0: loss = ([\d.]+)',new_txt)
data = []
x = []
for i in range(len(test_loss)):
    data.append(float(test_loss[i]))
    x.append(float(m[i]))
    
plt.plot(x, data)
plt.show()
