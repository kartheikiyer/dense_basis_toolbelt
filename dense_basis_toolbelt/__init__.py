from __future__ import print_function, division, absolute_import

__version__ = "0.0.1"
__bibtex__ = """
@article{iyer2019gpsfh,
  title={Non-parametric Star Formation History Reconstruction with Gaussian Processes I: Counting Major Episodes of Star Formation},
  author={Iyer, Kartheik G and Gawiser, Eric and Faber, Sandra M and Ferguson, Henry C and Koekemoer, Anton M and Pacifici, Camilla and Somerville, Rachel},
  journal={arXiv preprint arXiv:1901.02877},
  year={2019}
}
"""  # NOQA

from .samplers import *
from .sed_learning import *
