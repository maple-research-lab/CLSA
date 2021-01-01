# Publication:  "Protein Docking Model Evaluation by Graph Neural Networks", Xiao Wang, Sean T Flannery and Daisuke Kihara,  (2020)

#GNN-Dove is a computational tool using graph neural network that can evaluate the quality of docking protein-complexes.

#Copyright (C) 2020 Xiao Wang, Sean T Flannery, Daisuke Kihara, and Purdue University.

#License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

#Contact: Daisuke Kihara (dkihara@purdue.edu)

#

# This program is free software: you can redistribute it and/or modify

# it under the terms of the GNU General Public License as published by

# the Free Software Foundation, version 3.

#

# This program is distributed in the hope that it will be useful,

# but WITHOUT ANY WARRANTY; without even the implied warranty of

# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

# GNU General Public License V3 for more details.

#

# You should have received a copy of the GNU v3.0 General Public License

# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.

import os
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        print (path+" created")
        os.makedirs(path)
        return True
    else:
        print (path+' existed')
        return False
def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text