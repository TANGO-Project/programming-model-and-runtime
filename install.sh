#!/bin/bash -e

#Installs TANGO Programming Model and Runtime Abstraction layer

src_path=$(cd $(dirname $0) && pwd)
prefix=$1

shift 1

if [ -z "$prefix" ]; then
	echo " Installation prefix not specified"
	exit 1
fi

if [ -d "${prefix}" ]; then
	rm -rf ${prefix}
fi

# Build COMPSs 
echo " Building COMPS at ${prefix}/TANGO_ProgrammingModel/COMPSs from ${src_path}/COMPSs/"

mkdir -p ${prefix}/TANGO_ProgrammingModel/COMPSs
cd ${src_path}/COMPSs/builders

echo "./buildlocal $* ${prefix}/TANGO_ProgrammingModel/COMPSs "
./buildlocal $* ${prefix}/TANGO_ProgrammingModel/COMPSs
 
echo " Building OmpSs at ${prefix}/TANGO_ProgrammingModel/OmpSs from ${src_path}/OmpSs/"

mkdir -p ${prefix}/TANGO_ProgrammingModel/OmpSs

mkdir -p ${prefix}/TANGO_ProgrammingModel/OmpSs/nanox

cd ${src_path}/OmpSs/nanox
if [ -d "${prefix}/TANGO_ProgrammingModel/COMPSs/Dependencies/extrae/lib" ]; then
	echo "in $PWD running: autoreconf -fiv; autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox --with-extrae=${prefix}/TANGO_ProgrammingModel/COMPSs/Dependencies/extrae/ && make && make install"
	autoreconf -i && autoreconf -fiv; autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox --with-extrae=${prefix}/TANGO_ProgrammingModel/COMPSs/Dependencies/extrae/ && make && make install clean
else
	echo "in $PWD running: autoreconf -fiv; autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox && make && make install"
        autoreconf -i && autoreconf -fiv; autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox && make && make install clean
fi

mkdir -p ${prefix}/TANGO_ProgrammingModel/OmpSs/mcxx

cd ${src_path}/OmpSs/mcxx

echo "autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/mcxx --enable-ompss --with-nanox=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox/ && make && make install"

autoreconf -i && autoreconf -fiv; autoreconf -fiv && ./configure --prefix=${prefix}/TANGO_ProgrammingModel/OmpSs/mcxx --enable-ompss --with-nanox=${prefix}/TANGO_ProgrammingModel/OmpSs/nanox/ && make && make install clean
