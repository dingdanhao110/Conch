import numpy as np


path = "/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/"
out_file = "homograph"

label_file = "author_label"
PA_file = "PA"
PC_file = "PC"
PT_file = "PT"
APA_file = "APA"
APAPA_file = "APAPA"
APCPA_file = "APCPA"

PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                   dtype=np.int32)
PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                   dtype=np.int32)
PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                   dtype=np.int32)
PA[:, 0] -= 1
PA[:, 1] -= 1
PC[:, 0] -= 1
PC[:, 1] -= 1
PT[:, 0] -= 1
PT[:, 1] -= 1

paper_max = max(PA[:, 0]) + 1
author_max = max(PA[:, 1]) + 1
conf_max = max(PC[:, 1]) + 1
term_max = max(PT[:, 1]) + 1

PA[:, 0] += author_max
PC[:, 0] += author_max
PC[:, 1] += author_max+paper_max

edges = np.concatenate((PA,PC),axis=0)

np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')

