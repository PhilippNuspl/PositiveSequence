=======================================
PositiveSequence: A Mathematica package
=======================================

Introduction
=============

This package can be used to prove positivity of univariate C-finite and holonomic sequences.
To this end, methods from [GK05]_, [KP10]_, [PN22]_ and [OW14]_ are implemented.  

The package is developed by `Philipp Nuspl <mailto:philipp.nuspl@jku.at>`_ and
published under the GNU General Public License v3.0.
The research is funded by the 
Austrian Science Fund (FWF): W1214-N15, project DK15. 

Installation
============

The easiest way to install the package is simply to get a new version of ``RISCErgoSum``. 

RISCErgoSum
-----------

The package is part of ``RISCErgoSum`` which is a collection of Mathematica packages developed at RISC.
Instructions for the download and installation of ``RISCErgoSum`` can be found at:  

- `https://www3.risc.jku.at/research/combinat/software/ergosum/ <https://www3.risc.jku.at/research/combinat/software/ergosum/>`_

The demo notebook accompanying the ``RISCErgoSum`` version of the package can be obtained from

- `https://www.risc.jku.at/research/combinat/software/ergosum/RISC/PositiveSequence.html <https://www.risc.jku.at/research/combinat/software/ergosum/RISC/PositiveSequence.html>`_

GitHub
------

The package builds on the ``GeneratingFunctions`` package by Mallinger which is part of ``RISCErgoSum``. To use the ``PositiveSequence`` package, the ``GeneratingFunctions`` package has to be installed (hence, we recommend to simply install a new version of ``RISCErgoSum`` to also get ``PositiveSequence``). Then, the file ``PositiveSequence.m`` has to be put in a directory where Mathematica can find it. E.g., if it lies in the same directory as the worksheet for which it should be used, one can use package after executing (note that the ``GeneratingFunctions`` package has to be loaded first):

.. code:: Mathematica

    SetDirectory[NotebookDirectory[]];
    << RISC`GeneratingFunctions`;
    << PositiveSequence`;

Usage
======

Check out the Demo file on how ``PositiveSequence`` can be used (note that loading the package differs whether the package from GitHub or the RISCErgoSum package is loaded). The main method is called ``PositiveSequence``. You can check its documentation using ``?PositiveSequence``.

References
==========


.. [GK05] Stefan Gerhold, Manuel Kauers: A Procedure for Proving Special 
    Function Inequalities Involving a Discrete Parameter. In: Proceedings of 
    ISSAC'05, pp. 156–162. 2005. 

.. [KP10] Manuel Kauers, Veronika Pillwein: When can we detect that
    a P-finite sequence is positive? In: Proceedings of 
    ISSAC'10, pp. 195–202. 2010. 

.. [NP22] Philipp Nuspl, Veronika Pillwein: A comparison of algorithms for proving 
   positivity of linearly recurrent sequences. Submitted. 2022

.. [OW14] Joël Ouaknine and James Worrell. Positivity problems for low-order linear 
   recurrence sequences. In: Proceedings of SODA 14. 2014
