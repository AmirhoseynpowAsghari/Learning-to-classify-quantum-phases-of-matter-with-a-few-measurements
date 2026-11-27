"""qpom: Quantum Phases Of Matter -- a from-scratch reimplementation of the
methods in Khosrojerdi, Pereira, Cuccoli & Banchi,
"Learning to classify quantum phases of matter with a few measurements",
Quantum Sci. Technol. 10 (2025) 025006, arXiv:2404.18231.

This package reproduces the paper's two numerical experiments:

* ``qpom.annni``    -- the ANNNI spin chain (figures 1 and 2)
* ``qpom.haldane``  -- the cluster/Haldane SPT chain (figure 3)

and the shared machine-learning machinery:

* ``qpom.dmrg``     -- thin DMRG wrapper around quimb
* ``qpom.parity``   -- Z2 symmetry-sector projection (section 2.1)
* ``qpom.kernel``   -- squared-fidelity quantum kernel, k(x,y)=|<psi_x|psi_y>|^2
* ``qpom.svm``      -- kernel SVM classifier (appendix A) built on the kernel
* ``qpom.genbound`` -- appendix D generalization-error bound B = Tr sqrt(K)

No official code or data accompanies the paper (its data-availability
statement says "No new data were created or analysed in this study"), so
everything here is an independent reimplementation from the equations and
figures in the text.
"""
