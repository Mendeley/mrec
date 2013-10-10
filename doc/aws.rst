===================================
Running mrec on Amazon Web Services
===================================

If you have a large dataset of ratings, the SLIM recommender implemented here can take a fair number of CPU cycles to train because it has to solve a separate regression problem for each item.
Fortunately it's easy to reduce your waiting time by running in parallel on a cluster of computers
using the IPython.parallel framework.

The `StarCluster <https://github.com/jtriley/StarCluster>`_ project makes it extremely simple to 
provision an IPython cluster, by following the StarCluster `Quick-Start <>`_ and then
the instructions given `here <>`_.  To run `mrec` jobs on your cluster you'll need edit the `.starcluster/config` file to install the `mrec` package.  The cluster configuration should look
something like this:

.. code-block:: ini

    [cluster ip]
    KEYNAME = your-keypair
    CLUSTER_USER = ipuser
    NODE_IMAGE_ID = ami-6c3a2f18
    NODE_INSTANCE_TYPE = m1.xlarge
    CLUSTER_SIZE = 40
    CLUSTER_SHELL = bash
    DISABLE_QUEUE = True
    SPOT_BID = 0.15
    PLUGINS = python-packages, ipcluster
    VOLUMES = your-s3-volume

    [plugin python-packages]
    setup_class = starcluster.plugins.pypkginstaller.PyPkgInstaller
    install_command = pip install -U %s
    packages = pyzmq,
               git+http://github.com/ipython/ipython.git,
               mrec

    [plugin ipcluster]
    SETUP_CLASS = starcluster.plugins.ipcluster.IPCluster
    PACKER = pickle
    ENABLE_NOTEBOOK = True

This specifies an ``ip`` cluster template based on a StarCluster Ubuntu image which already has
a number of scientific Python libraries installed.  The template also specifies two plugins
to run after the machines are booted.  The first of these installs the remaining required Python
packages: pyzmq, the latest version of IPython from github (this can be a good idea but but your mileage may vary), and mrec itself.  Finally the second plugin launches the IPython controller and worker processes themselves, and specifies ``pickle`` as the packer used to serialize objects
passed between them.

You can then fire up a cluster ready to run `mrec` jobs::

    $ starcluster start -c ip mrec_cluster

This launches the number of nodes specified in the ``ip`` template, starts a controller on the
master node and a worker process on each remaining core.  It also sets up a shared NFS file system
visible to all of the nodes.

You can make your training data available either on an s3 volume, by following the instructions
in the StarCluster documentation (usually just by configuring it in the StarCluster config file)
or by putting it to the NFS by hand like this::

    $ starcluster sshmaster -u ipuser mrec_cluster 'mkdir data'
    $ starcluster put -u ipuser /path/to/datasets data/

