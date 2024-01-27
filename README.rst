SDOstream-c
=======

SDOstream-c, is a powerful clustering algorithm designed specifically for clustering dynamic and continuous data streams. Built upon and implemented on top of the efficient `dSalmon <https://https://github.com/CN-TU/dSalmon>`_ framework.


(Re-)Building
^^^^^^^^^^

To simplify the build process and ensure consistency, you can use the provided sdo_env.yml file to create a conda environment. This environment includes Python dependencies necessary for building the framework and running experiments.

.. code-block:: sh

    conda env create -f sdo_env.yml


When adding new algorithms or modifying the interface, the SWIG wrappers have to be rebuilt. To this end, SWIG has to be installed and a ``pip`` package can be created and installed  using

.. code-block:: sh

    make && pip3 install dSalmon.tar.xz


Experiments
^^^^^^^^^^

The experiments utilizes the data and scripts from `py-temporal-silhouette <https://github.com/CN-TU/py-temporal-silhouette>`_

Install `DenStream<https://github.com/issamemari/DenStream>_` by adding the files *DenStream.py* and *MicroCluster.py* to the [/tests/temporal-silhouette/] folder.

For the synthetic data, open a terminal in the [/tests/temporal-silhouette/] folder. Run:

.. code-block:: sh

    python3 run_analysis_synthetic.py

A *results_synthetic.csv* file will be created in the [/tests/temporal-silhouette/] folder. 

For the real data, open a terminal in the [/tests/temporal-silhouette/] folder. Run:

.. code-block:: sh 

    python3 run_analysis_real.py

Three files (*results_real.csv/tex*, *fert_vs_gdp_labels.csv* and *retail_labels.csv*) will be created in the [results/] folder.

For the flow data (subsample of `TII-SSRC-23 Dataset <https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23>_`), open a terminal in the [/tests/temporal-silhouette/] folder. Run:

.. code-block:: sh 

    python3 run_analysis_flow.py

Four files (*results_flow.csv/tex*, *flow_traffic_type_labels.csv* and *flow_traffic_subtype_labels.csv*) will be created in the [results/] folder.