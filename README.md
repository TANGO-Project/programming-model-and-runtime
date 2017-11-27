# TANGO Programming Model and Runtime Abstraction Layer
&copy; Barcelona Supercomputing Center 2016 -2017

## Description

The TANGO Programming Model and Runtime Abstraction Layer is a combination of the BSC's COMPSs and OmpSs task-based programming models, where COMPSs is dealing with the coarse-grain tasks and platform-level heterogeneity and OmpSs is dealing with fine-grain tasks and node-level heterogeneity.

## Installation Guide

### Dependencies:
#### Common
- Supported platforms running Linux (i386, x86-64, ARM, PowerPC or IA64)
- Apache maven 3.0 or better
- Java SDK 8.0 or better
- GNU C/C++ compiler versions 4.4 or better
- CNU GCC Fortran
- autotools (libtool, automake, autoreconf, make) 
- boost-devel
- python-devel 2.7 or better
- GNU bison 2.4.1 or better.
- GNU flex 2.5.4 or 2.5.33 or better. (Avoid versions 2.5.31 and 2.5.34 of flex as they are known to fail. Use 2.5.33 at least.)
- GNU gperf 3.0.0 or better. 
- SQLite 3.6.16 or better. 


#### --with-monitor option:
- xdg-utils package 
- graphviz package

#### --with-tracing option
- libxml2-devel 2.5.0 or better 
- gcc-fortran
- papi-devel (sugested)

### Installation script

To install the whole framework you just need to checkout the code and run the following command

```bash
$ git clone --recursive https://github.com/TANGO-Project/programming-model-and-runtime.git

$ cd programming-model-and-runtime/

$ ./install.sh <Installation_Prefix> [options]

#Examples

#User local installation

$./install.sh $HOME/TANGO --no-monitor --no-tracing

#System installation

$ sudo -E ./install.sh /opt/TANGO
```

## User Guide

### Application Development Overview

To develop an application with the TANGO programming model, developers has to at least implement 3 files: the application main workflow in appName.c/cc, the application functions which are going to be coarse-grain tasks in appName.idl, and the implementation of the functions in appName-functions.cc. Other application files can be included in a src folder providing the building configuration in a Makefile   

- appName.c/cc -> Contains the main coarse-grain task workflow
- appName.idl -> Coarse-grain task definition
- appName-functions.c/cc -> Implementation of the coarse grain tasks

To define a coarse-grain task which contains fine-grain tasks, developers have to annotate the coarse-grain fucntions with the OmpSs pragmas. 

More information about how to define coarse-grain tasks and other concerns when implementing a coarse-grain task workflow can be found in http://compss.bsc.es/releases/compss/latest/docs/COMPSs_User_Manual_App_Development.pdf

More information about how to define fine-grain tasks and other considerations when implementing a fine-grain task workflow can be found in https://pm.bsc.es/ompss-docs/specs/

### Application Compilation

To compile the application use the compss_build_app script. The usage of this command is the following

```bash
$ compss_build_app --help
Usage: /opt/COMPSs/Runtime/scripts/user/compss_build_app [options] application_name

* Options:
  General:
    --help, -h                              Print this help message

    --opts                                  Show available options

    --version, -v                           Print COMPSs version

  Tools enablers:
    --ompss                                 Enables worker compilation with OmpSs Mercurium compiler
                                            Default: Disabled
    --cuda                                  Enables worker compilation with CUDA compilation flags
                                            Default: Disabled
    --opencl                                Enables worker compilation with openCL compilation flags
                                            Default: Disabled
    --with_ompss=<ompss_installation_path>  Enables worker compilation with OmpSs Mercurium compiler installed in a certain location
                                            Default: Disabled
    --mercurium_flgas="flags"               Specifies extra flags to pass to the mercurium compiler
                                            Default: Empty
    --with_cuda=<cuda_installation_path>    Enables worker compilation with CUDA installed in a certain location
                                            Default: Disabled
    --with_opencl=<ocl_installation_path>   Enables worker compilation with openCL installed in a certain location
                                            Default: Disabled

```
To compile with only COMPSs application use the following command:

```bash
$ compss_build_app appName
```
To compile with a COMPSs application with fine-grain managed by OmpSs tasks use the following command:

```bash
$ compss_build_app --ompss appName
```

If there are fine-grain tasks which can be run in CUDA devices use the following command:

```bash
$ compss_build_app --ompss --cuda appName
```

### Application Execution

An application implemented with the TANGO programming model can be easily executed by using the COMPSs execution scripts. It will automatically starts the Runtime Abstraction Layer and execute transparently either coarse-grain and fine-grain tasks in the selected resources. 

Users can use the runcompss script to run the application in interactive nodes.

```bash
Usage: runcompss [options] application_name application_arguments  
```

An example to run the application in the localhost (interesting for initial debugging)

```bash
$ runcompss --lang=c appName appArgs...
```

To run an application in a preconfigured grid of computers you have to provide the resource description in a resources.xml file and the application configuration in these resources in the project.xml. Information about how to define this files can be found in http://compss.bsc.es/releases/compss/latest/docs/COMPSs_User_Manual_App_Exec.pdf

```bash
runcompss --lang=c --project=/path/to/project.xml --resources=/path/to/resources.xml appName app_args
```

More information about other possible arguments can be found by executing

```bash
$ runcompss --help       
``` 

To queue an application in a cluster managed by the SLURM resource manager, users has to use the enqueue_compss command.

```bash
Usage: enqueue_compss [queue_system_options] [rucompss_options] application_name application_arguments
```

The following command show how to queue the application by requesting 3 nodes with at least 12 cores, 2 gpus and 32GB of memory (approx.)

```bash
$ enqueue_compss --num_nodes=3 --cpus-per-node=12 --gpus-per-node=2 --node-memory=32000 --lang=c appName appArgs
```
Other options available for enqueue_compss can be found by executing 

```bash
$ enqueue_compss --help
```
## Relation to other TANGO components

TANGO Programming Model and the Runtime Abstraction Layer components must be used together as explained above. It can be combined with other TANGO components to achive more complex features:

* **ALDE** - TANGO Programming Model and Runtime will use ALDE to submit the code for compilation and packetization. Also it could be intereact with ALDE to submit the application directly to a TANGO compatible device supervisor.
* **Device Supervisor** -  The TANGO Programming Model Runtime can interact directly with SLURM which is one TANGO Device Supervisor. Other device supervisors will be supported by means of ALDE. 
* **Self-Adaptation Manager** - The TANGO Programming Model Runtime will interact with the Self-Adaptation Manager to optimize application execution in a TANGO compatible testbed.
* **Monitor Infrastructure** - The TANGO Programming Model Runtime will interact with TANGO Monitor Infrastructure to obtain the energy consumption metrics in order to take runtime optimization decisions.
