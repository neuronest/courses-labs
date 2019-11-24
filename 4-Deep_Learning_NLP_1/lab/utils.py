# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>


def arrow(idx_batch, nb_batch, size_arrow=50):
    x = 1 / nb_batch * idx_batch * size_arrow
    x = int(x + 1) if x != int(x) else int(x)
    return "=" * x + ">" + (size_arrow - x) * "."
