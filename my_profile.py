"""HCL's CloudLab Profile"""

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

# Create a portal object,
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# Variable number of nodes.
pc.defineParameter("nodeCount", "Number of Nodes", portal.ParameterType.INTEGER, 1,
                   longDescription="If you specify more then one node, " +
                   "we will create a lan for you.")

# Pick your OS.
imageList = [
    ('urn:publicid:IDN+wisc.cloudlab.us+image+lb-benchmark-PG0:moeUpdate1', 'Default Image'),
    ('urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-STD', 'UBUNTU 18.04'),
    ('urn:publicid:IDN+wisc.cloudlab.us+image+nestfarm-PG0:HCL-U20', 'HCL U20.04'),
    ('urn:publicid:IDN+wisc.cloudlab.us+image+nestfarm-PG0:HCL-U2204', 'HCL U22.04'), # Ubuntu 22.04.1 Linux kernel 6.0
    ('urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD', 'UBUNTU 20.04'),
    ('urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU16-64-STD', 'UBUNTU 16.04'),
    ('urn:publicid:IDN+emulab.net+image+emulab-ops//CENTOS7-64-STD',  'CENTOS 7'),
    ('urn:publicid:IDN+emulab.net+image+emulab-ops//FBSD112-64-STD', 'FreeBSD 11.2')]

pc.defineParameter("osImage", "Select OS image",
                   portal.ParameterType.IMAGE,
                   imageList[3], imageList,
                   longDescription="Most clusters have this set of images, " +
                   "pick your favorite one.")

# Optional physical type for all nodes.
pc.defineParameter("phystype",  "Optional physical node type",
                   portal.ParameterType.STRING, "c220g5",
                   longDescription="Specify a physical node type (m510,d710,etc) " +
                   "instead of letting the resource mapper choose for you.")

# Optionally create XEN VMs instead of allocating bare metal nodes.
pc.defineParameter("useVMs",  "Use XEN VMs",
                   portal.ParameterType.BOOLEAN, False,
                   longDescription="Create XEN VMs instead of allocating bare metal nodes.")

# Optional link speed, normally the resource mapper will choose for you based on node availability
pc.defineParameter("linkSpeed", "Link Speed",portal.ParameterType.INTEGER, 0,
                   [(0,"Any"),(100000,"100Mb/s"),(1000000,"1Gb/s"),(10000000,"10Gb/s"),(25000000,"25Gb/s"),(100000000,"100Gb/s")],
                   advanced=True,
                   longDescription="A specific link speed to use for your lan. Normally the resource " +
                   "mapper will choose for you based on node availability and the optional physical type.")

# For very large lans you might to tell the resource mapper to override the bandwidth constraints
# and treat it a "best-effort"
pc.defineParameter("bestEffort",  "Best Effort", portal.ParameterType.BOOLEAN, False,
                    advanced=True,
                    longDescription="For very large lans, you might get an error saying 'not enough bandwidth.' " +
                    "This options tells the resource mapper to ignore bandwidth and assume you know what you " +
                    "are doing, just give me the lan I ask for (if enough nodes are available).")

# Optional ephemeral blockstore
pc.defineParameter("tempFileSystemSize", "Temporary Filesystem Size",
                   portal.ParameterType.INTEGER, 0,advanced=True,
                   longDescription="The size in GB of a temporary file system to mount on each of your " +
                   "nodes. Temporary means that they are deleted when your experiment is terminated. " +
                   "The images provided by the system have small root partitions, so use this option " +
                   "if you expect you will need more space to build your software packages or store " +
                   "temporary files.")

# Instead of a size, ask for all available space.
pc.defineParameter("tempFileSystemMax",  "Temp Filesystem Max Space",
                    portal.ParameterType.BOOLEAN, False,
                    advanced=True,
                    longDescription="Instead of specifying a size for your temporary filesystem, " +
                    "check this box to allocate all available disk space. Leave the size above as zero.")

pc.defineParameter("tempFileSystemMount", "Temporary Filesystem Mount Point",
                   portal.ParameterType.STRING,"/tdata",advanced=True,
                   longDescription="Mount the temporary file system at this mount point; in general you " +
                   "you do not need to change this, but we provide the option just in case your software " +
                   "is finicky.")

# Retrieve the values the user specifies during instantiation.
params = pc.bindParameters()

# Check parameter validity.
if params.nodeCount < 1:
    pc.reportError(portal.ParameterError("You must choose at least 1 node.", ["nodeCount"]))

if params.tempFileSystemSize < 0 or params.tempFileSystemSize > 200:
    pc.reportError(portal.ParameterError("Please specify a size greater then zero and " +
                                         "less then 200GB", ["nodeCount"]))
pc.verifyParameters()

# Create link/lan.
if params.nodeCount > 1:
    if params.nodeCount == 2:
        lan = request.Link()
    else:
        lan = request.LAN()
        pass
    if params.bestEffort:
        lan.best_effort = True
    elif params.linkSpeed > 0:
        lan.bandwidth = params.linkSpeed
    pass

# Process nodes, adding to link or lan.
for i in range(params.nodeCount):
    # Create a node and add it to the request
    if params.useVMs:
        name = "vm" + str(i)
        node = request.XenVM(name)
    else:
        name = "node" + str(i)
        node = request.RawPC(name)

        if False: # HCL: persistent dataset (disabled)
            # Connect to the persistent dataset (remote block storage)
            iface = node.addInterface()
            # The remote file system is represented by special node.
            fsnode = request.RemoteBlockstore("fsnode", "/pdata")
            # This URN is displayed in the web interfaace for your dataset.
            #fsnode.dataset = "urn:publicid:IDN+wisc.cloudlab.us:nestfarm-pg0+ltdataset+ioda-vm-image"
            #fsnode.dataset = "urn:publicid:IDN+wisc.cloudlab.us:nestfarm-pg0+ltdataset+ioda-vm-image2"
            #fsnode.dataset = "urn:publicid:IDN+wisc.cloudlab.us:nestfarm-pg0+ltdataset+ioda-vm-image3"
            #fsnode.dataset = "urn:publicid:IDN+wisc.cloudlab.us:nestfarm-pg0+ltdataset+ioda-vm-image4"
            fsnode.dataset = "urn:publicid:IDN+wisc.cloudlab.us:nestfarm-pg0+ltdataset+p5"
            fslink = request.Link("fslink")
            fslink.addInterface(iface)
            fslink.addInterface(fsnode.interface)
            # Special attributes for this link that we must use.
            fslink.best_effort = True
            fslink.vlan_tagging = True
        pass
    if params.osImage and params.osImage != "default":
        node.disk_image = params.osImage
        pass
    # Add to lan
    if params.nodeCount > 1:
        iface = node.addInterface("eno1")
        iface.addAddress(pg.IPv4Address("192.168.1." + str(i), "255.255.255.0"))
        lan.addInterface(iface)
        pass
    # Optional hardware type.
    if params.phystype != "":
        node.hardware_type = params.phystype
        pass
    # Optional Blockstore
    if params.tempFileSystemSize > 0 or params.tempFileSystemMax:
        bs = node.Blockstore(name + "-bs", params.tempFileSystemMount)
        if params.tempFileSystemMax:
            bs.size = "0GB"
        else:
            bs.size = str(params.tempFileSystemSize) + "GB"
            pass
        bs.placement = "any"
        pass
    pass

    # # HCL: ``setup-host.sh`` will perform initial settings to the cloudlab node
    # node.addService(pg.Execute(shell = "bash", command = '/local/repository/silly.sh'))

# Print the generated rspec
pc.printRequestRSpec(request)
