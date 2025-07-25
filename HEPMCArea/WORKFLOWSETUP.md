# Workflow 

This takes you through how to setup a workflow for analysing HEPMC files in Rivet, and assumes that you already have your desired HEPMC file, from opendata or elsewhere. 

This contains three main components 
    * HEPMC
    * Pythia
    * Rivet

This also assumes working from CERNs based lxplus cluster.


## HEPMC 

Go to the [HEPMC website](http://hepmc.web.cern.ch/hepmc/)

Dowload v2.06.11 and unpack: tar -xzvf {your_file_name}.tg

In the command line run these commands in order.

cd HepMC-2.06.11

./configure --prefix=$PWD/../local --with-momentum=GEV --with-length=MM #(hepmc3 is ./configure --prefix=$PWD/local)

make

make install

## Pythia

Go to the [Pythia website](http://home.thep.lu.se/Pythia/)

Dowload v8.2.4.4 and unpack: tar -xzvf {your_file_name}.tg

cd pythia82444

./configure --with-hepmc2=<path of HepMC>/local


## Rivet

Run the following commands in order 

> setupATLAS

> lsetup asetup

> asetup 23.6.26,AthGeneration [or you can try:]> asetup AthGeneration, 23.6.26

> source setupRivet 


To plot in rivet 

rivet {--skip-weights} -a $CONTUR_RA13TeV -o {output_name}.yoda {input_name}.hepmc.gz  -> works on just .hepmc files aswell

Rivet-mkhtml {output_name}.yoda

This will create a folder with .pngs created from the anaysis used. 