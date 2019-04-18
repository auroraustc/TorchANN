# Torch-NNMD
Use torch to train NN potential

## Features
* Supports [DeePMD](https://github.com/deepmodeling/deepmd-kit) types of symmetry coordinates.
* Trains on multi-gpus.

## Request:
1. python version >= 3
2. [pyTorch](https://pytorch.org/get-started/locally/) (necessary: Torch; optional: Torchvision)
3. C compiler (icc 2018 tested)

## Prepare training data:
```bash
box.raw type.raw coord.raw force.raw energy.raw
```
Property| Unit
---	| :---:
Time	| ps
Length	| Å
Energy	| eV
Force	| eV/Å
Pressure| Bar

#### `box.raw` contains box of each frame:
```bash
 7.710900  0.000000  0.000000  0.000000  8.904000  0.000000  0.000000  0.000000 26.296000 
 7.710900  0.000000  0.000000  0.000000  8.904000  0.000000  0.000000  0.000000 26.296000 
 7.710900  0.000000  0.000000  0.000000  8.904000  0.000000  0.000000  0.000000 26.296000  
 7.667900  0.000000  0.000000 -11.501850 19.921789  0.000000  0.000000  0.000000 21.260799 
 7.667900  0.000000  0.000000 -11.501850 19.921789  0.000000  0.000000  0.000000 21.260799 
 7.667900  0.000000  0.000000 -11.501850 19.921789  0.000000  0.000000  0.000000 21.260799
 ...
```
Each line contains nine numbers: a1, a2, a3; b1, b2, b3; c1, c2, c3.
Take the first line as an exampe:
```bash
 7.710900  0.000000  0.000000  0.000000  8.904000  0.000000  0.000000  0.000000 26.296000
```
means a box of:
```bash
vec_a = (7.710900, 0.000000, 0.000000)
vec_b = (0.000000, 8.904000, 0.000000)
vec_c = (0.000000, 0.000000,26.296000)
```
#### `type.raw` contains type of each atom in one frame:
```bash
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 
```
Here `0` and `1` represent two different types of atoms. The numbers function as labels of different element, therefore the values could be arbitary.

#### `coord.raw` stores the coordinates of atoms in one frame:
```bash
 2.939720  6.538850  2.954150  1.850690  6.321660  0.736290  4.276000  8.778080  2.631220  4.612180  7.558540  4.802170  0.551250  8.550070  0.733660  0.655030  6.664390  2.937170  3.113080  0.419850  5.534930  3.123380  8.568410  0.743650  0.778900  0.453260  5.211400  4.408090  6.334930  0.735760  1.849100  0.016200  2.721580  1.857600  6.967150  5.332680  4.405930  1.883460  0.735760  3.121450  4.118100  0.742600  2.949370  2.374500  2.578150  0.194690  2.338610  3.065110  3.308660  5.091720  4.742190  4.441900  4.275650  2.786670  4.261720  3.017710  5.301710  1.778140  4.519680  2.751790  1.900780  2.518370  4.560860  0.914160  4.823890  4.846900  0.555570  4.098240  0.733130  1.845300  1.871980  0.735760  6.991240  6.321660  0.736290  5.691800  8.550070  0.733660  5.741350  6.848900  2.586650  7.341340  7.182960  5.095860  6.951990  0.137590  3.165190  0.294910  5.856450  7.814560  5.704330  2.236160  2.999860  7.062350  4.585110  2.871480  6.699420  2.917990  5.042990  5.583340  4.966420  5.114320  5.696120  4.098240  0.733130  6.985840  1.871980  0.735760  4.022520  7.317680  7.556660  6.571000  7.409080  7.691360  3.024180  0.713790  7.936110  4.204900  8.777400  7.712050  4.180610  2.885990  7.801380  5.499700  0.609370  7.984420  6.806580  8.902100  7.918360  6.806870  2.871500  7.769460  2.977010  2.095440  7.859100  2.948980  5.082750  7.984330  4.150370  4.408660  7.961870  5.495450  2.058400  7.841980  5.390350  5.021750  7.704490  6.816330  4.345000  7.692350  2.911550  6.479290  7.755130  5.447710  6.499890  7.516550  1.665270  7.231910  7.824960  0.419990  0.751450  7.927650  1.528080  8.637960  8.123200  1.703230  2.980040  8.090160  0.454800  2.287580  7.972300  1.807610  4.392440  8.106240 
 3.120500  6.994950  2.949000  1.850690  6.321660  0.736290  4.195500  0.137570  3.207200  4.359900  7.345810  5.352240  0.551250  8.550070  0.733660  0.161560  7.125580  3.070050  3.496400  0.684930  5.298330  3.123380  8.568410  0.743650  0.651680  0.820960  4.879630  4.408090  6.334930  0.735760  1.777200  0.238560  2.589990  1.910160  7.465450  5.420560  4.405930  1.883460  0.735760  3.121450  4.118100  0.742600  3.216250  2.589190  3.767250  0.516070  2.320130  2.712070  2.682760  5.240930  5.003450  4.106400  4.743610  2.826160  4.325630  3.457960  5.460560  1.746680  4.366620  2.877730  1.389450  3.361190  5.324870  0.066800  5.167380  5.431660  0.555570  4.098240  0.733130  1.845300  1.871980  0.735760  6.991240  6.321660  0.736290  5.691800  8.550070  0.733660  5.623340  7.049830  3.068470  6.839820  7.282070  5.414690  6.448320  0.099900  3.361960  0.596430  5.570910  7.522860  5.529760  2.331660  2.991950  7.202050  4.575950  2.920380  6.762590  2.744450  5.309250  5.540470  5.169300  5.068410  5.696120  4.098240  0.733130  6.985840  1.871980  0.735760  4.679570  7.015460  7.830100  7.105320  6.868540  7.626400  3.232950  0.216650  7.622630  4.574330  8.473800  7.798490  4.502490  2.408260  7.686960  5.808430  0.459780  7.765280  7.073010  8.438140  7.512430  7.079220  2.716440  7.780310  3.210720  1.677380  7.711180  3.204360  4.859500  8.017850  4.496510  3.935950  7.855960  5.835410  1.844390  7.614220  5.701180  4.715300  7.780400  6.851490  4.166810  7.765460  3.057940  6.288290  7.714330  5.809890  6.177390  7.881030  1.886420  7.124600  7.360930  0.547770  0.443140  7.540130  1.820000  8.495150  7.339720  1.912260  2.545990  7.607010  0.606200  1.864010  7.730840  2.047170  4.021280  7.747450 
 3.112030  7.005310  2.958420  1.850690  6.321660  0.736290  4.189430  0.128210  3.194720  4.357020  7.343710  5.359230  0.551250  8.550070  0.733660  0.160430  7.126910  3.086360  3.499650  0.679880  5.304050  3.123380  8.568410  0.743650  0.640140  0.834520  4.879190  4.408090  6.334930  0.735760  1.785940  0.238440  2.596630  1.895200  7.473770  5.416660  4.405930  1.883460  0.735760  3.121450  4.118100  0.742600  3.205240  2.606860  3.791300  0.516330  2.322430  2.715500  2.677730  5.251680  5.009540  4.101450  4.751660  2.824440  4.341590  3.471110  5.455930  1.747990  4.374300  2.875750  1.383420  3.370110  5.336510  0.066550  5.169330  5.418750  0.555570  4.098240  0.733130  1.845300  1.871980  0.735760  6.991240  6.321660  0.736290  5.691800  8.550070  0.733660  5.618360  7.051110  3.069130  6.842410  7.290190  5.422980  6.467830  0.098520  3.363100  0.611390  5.577910  7.522050  5.535000  2.348870  2.988000  7.195520  4.571400  2.917060  6.762370  2.742990  5.303560  5.528970  5.146850  5.070090  5.696120  4.098240  0.733130  6.985840  1.871980  0.735760  4.677550  7.015900  7.832900  7.130750  6.857830  7.594210  3.199580  0.202120  7.625500  4.572570  8.443950  7.840930  4.510280  2.369360  7.694390  5.813740  0.444550  7.757600  7.078400  8.421720  7.533060  7.091530  2.705350  7.770630  3.191240  1.681540  7.718300  3.233580  4.841050  7.995790  4.481970  3.915460  7.869820  5.815950  1.851240  7.576770  5.698690  4.722620  7.757330  6.879610  4.164200  7.754710  3.058440  6.288870  7.685320  5.797690  6.189200  7.832260  1.882780  7.107990  7.338960  0.525580  0.449370  7.528100  1.854690  8.507680  7.343310  1.954980  2.546380  7.588540  0.592690  1.857680  7.716280  2.056580  4.034160  7.787930 
 1.282800  2.212310  0.039760 -8.144420 19.347740  4.085240  6.215390  0.757420  1.897690  3.841620  2.212120  0.038910 -10.750040 19.272430  3.795100  1.103990  0.574690  1.689960 -1.272530  2.213110  0.036990  2.237010  0.064850  6.143520  3.735900  0.970690  2.226780  0.001760  4.428220  0.027430  1.882580  1.743410  3.902960 -0.093660  2.609750  2.402980  2.561270  4.428810  0.039120  5.490550  0.553690  4.235460  2.989180  3.294010  2.286810 -2.546050  4.428610  0.028060 -0.011330  1.730800  5.377530  5.376480  2.921220  1.833860 -1.271150  6.635350  0.028910  1.240240  4.054270  4.357950 -2.588110  5.438170  1.959160  1.283760  6.639930  0.036780  4.011410  4.654690  4.378530  0.143110  5.240130  2.399690 -3.828310  6.640530  0.044220 -1.597310  3.576900  4.136180  2.475790  5.723850  2.142770 -2.551110  8.852840  0.039760  0.284490  6.478610  4.392990 -3.646020  7.961560  2.274960  0.007590  8.852840  0.038910  2.681140  7.060980  4.030470 -1.067040  7.859500  2.231120 -5.106550  8.853840  0.036990 -2.096530  6.922060  4.126020  1.448760  8.301170  1.929250 -3.832150 11.068750  0.027430 -5.048350  9.394130  5.193820 -5.223310  9.948230  2.906820 -1.272640 11.069340  0.039120 -2.544470  9.557530  4.570610 -2.395560 10.660440  2.501560 -6.379960 11.069140  0.028060  0.526200  8.691680  4.205300 -0.299950 10.555280  2.270450 -5.105060 13.275880  0.028910 -5.682980 12.034100  4.572380 -6.127530 12.270750  2.124080 -2.550150 13.280460  0.036780 -3.199250 12.132430  4.255530 -4.033210 13.526430  2.371890 -7.662230 13.281060  0.044220 -0.608960 11.823160  4.577450 -1.622170 12.987430  2.335600 -6.385020 15.493370  0.039760 -6.075000 16.122790  6.224120 -5.810030 16.161570  2.301890 -3.826320 15.493370  0.038910 -6.892850 13.892490  6.422450 -4.842590 14.905860  4.223450 -8.940470 15.494370  0.036990 -7.352800 13.806640  3.893940 -2.241940 14.801190  4.087140 -7.666170 17.709470  0.027430 -5.112170 17.253040  4.365730 -8.351570 15.335220  2.393300 -5.106670 17.710070  0.039120 -3.938600 16.631870  6.654630 -3.551740 15.925770  2.191850 -10.213990 17.709870  0.028060 -8.542710 15.633880  6.129220 -2.371150 17.300640  4.789790 -8.939080 19.916610  0.028910 -7.364190 17.125720  3.924720 -9.548590 17.661200  2.565380 -6.384180 19.921190  0.036780 -4.191650 18.890540  5.880700 -7.126000 18.479830  2.164090 -11.496140 19.921590  0.044220 -7.801300 17.994070  6.020740 -4.499310 18.262000  2.262530 -3.650700 13.958980  6.649480 -4.877950 14.312180  6.939930 -1.296120 13.068070  6.502380 -2.249570 13.906730  6.584870  0.084850 10.835580  6.731200 -0.166780 12.147960  6.408330  0.441670  8.388820  6.044280  0.663110  9.657400  6.417660 -1.409550  6.299840  5.961230 -0.322650  7.146510  6.197500 -0.571820  3.610700  5.835240 -1.068860  4.868430  6.011910 
 1.282800  2.212310  0.039760 -8.148690 19.344870  4.070540  6.225990  0.756000  1.905160  3.841620  2.212120  0.038910 -10.747680 19.264990  3.791540  1.102660  0.556370  1.692410 -1.272530  2.213110  0.036990  2.250140  0.057370  6.140490  3.745820  0.971250  2.229880  0.001760  4.428220  0.027430  1.898430  1.737300  3.897810 -0.083840  2.602610  2.400730  2.561270  4.428810  0.039120  5.487000  0.538500  4.236090  2.992080  3.294560  2.276170 -2.546050  4.428610  0.028060 -0.010970  1.742640  5.377260  5.377950  2.925160  1.820120 -1.271150  6.635350  0.028910  1.238590  4.037450  4.347850 -2.585940  5.438230  1.939110  1.283760  6.639930  0.036780  4.012150  4.672020  4.370880  0.151550  5.245380  2.397580 -3.828310  6.640530  0.044220 -1.604380  3.580240  4.141270  2.471720  5.732290  2.143790 -2.551110  8.852840  0.039760  0.288490  6.477280  4.379320 -3.634390  7.953540  2.268980  0.007590  8.852840  0.038910  2.689060  7.059320  4.015330 -1.064390  7.861530  2.225750 -5.106550  8.853840  0.036990 -2.089110  6.921490  4.128250  1.443540  8.298490  1.939770 -3.832150 11.068750  0.027430 -5.050900  9.391510  5.208850 -5.229770  9.951400  2.873830 -1.272640 11.069340  0.039120 -2.555880  9.552630  4.586930 -2.400290 10.658570  2.509890 -6.379960 11.069140  0.028060  0.540580  8.688430  4.193650 -0.284530 10.547230  2.264770 -5.105060 13.275880  0.028910 -5.685160 12.048420  4.583060 -6.134660 12.275460  2.125300 -2.550150 13.280460  0.036780 -3.195580 12.135820  4.247110 -4.030020 13.536130  2.380980 -7.662230 13.281060  0.044220 -0.612330 11.811200  4.557520 -1.623020 12.993830  2.339420 -6.385020 15.493370  0.039760 -6.062170 16.127660  6.214140 -5.803260 16.166880  2.310440 -3.826320 15.493370  0.038910 -6.906190 13.884100  6.411420 -4.831400 14.908190  4.230400 -8.940470 15.494370  0.036990 -7.345330 13.797330  3.893770 -2.216480 14.790410  4.093970 -7.666170 17.709470  0.027430 -5.111920 17.252190  4.371340 -8.349100 15.331590  2.411990 -5.106670 17.710070  0.039120 -3.935930 16.646930  6.667140 -3.558240 15.916780  2.187180 -10.213990 17.709870  0.028060 -8.554650 15.641850  6.121360 -2.372310 17.306310  4.797400 -8.939080 19.916610  0.028910 -7.349680 17.129930  3.935300 -9.538110 17.652070  2.577920 -6.384180 19.921190  0.036780 -4.194950 18.897160  5.893140 -7.143260 18.488880  2.164200 -11.496140 19.921590  0.044220 -7.803960 18.005790  6.012380 -4.509860 18.277330  2.272730 -3.645530 13.984950  6.631980 -4.857030 14.309520  6.972420 -1.287960 13.066950  6.466180 -2.275770 13.882910  6.566400  0.078050 10.822940  6.745460 -0.158960 12.147450  6.370870  0.469150  8.408800  6.060470  0.670800  9.630560  6.429520 -1.427280  6.276960  5.974540 -0.293150  7.132910  6.206960 -0.546530  3.599010  5.807840 -1.074520  4.877300  6.014470 
 1.282800  2.212310  0.039760 -8.230690 19.019460  4.191020  6.688540  1.229780  2.693880  3.841620  2.212120  0.038910 -11.205490 19.600690  3.884580 -9.725040 19.901900  2.167700 -1.272530  2.213110  0.036990  1.796650  0.427850  5.898650  4.361010  1.171280  2.140280  0.001760  4.428220  0.027430  2.812280  1.870350  4.104780  0.670480  2.594600  2.072830  2.561270  4.428810  0.039120 -6.024260 19.145710  4.453130  2.900640  3.249350  2.133160 -2.546050  4.428610  0.028060 -0.058550  1.971560  4.988910  5.294730  3.477730  2.036330 -1.271150  6.635350  0.028910  1.592160  4.147660  4.050620 -2.010610  5.996470  2.375830  1.283760  6.639930  0.036780  3.655260  4.968680  3.821450  0.002330  5.093610  2.337150 -3.828310  6.640530  0.044220 -1.709770  3.989360  4.486400  2.658990  5.746690  1.786200 -2.551110  8.852840  0.039760  1.141990  6.351940  3.909630 -3.866510  8.105690  2.098420  0.007590  8.852840  0.038910  2.814710  8.078220  4.143360 -1.042550  8.262650  2.423140 -5.106550  8.853840  0.036990 -1.836460  7.012820  4.681370  1.499740  7.877290  2.236170 -3.832150 11.068750  0.027430 -5.320420  9.393190  6.032720 -5.007720 11.017230  2.172680 -1.272640 11.069340  0.039120 -3.036320  9.463990  4.399410 -2.653480 10.103650  2.156260 -6.379960 11.069140  0.028060  1.043820  9.575860  3.994380  0.773510  9.888170  1.845220 -5.105060 13.275880  0.028910 -5.345270 12.049380  4.636270 -6.640890 12.660660  1.901410 -2.550150 13.280460  0.036780 -2.984460 12.045600  4.173510 -4.168720 13.361730  2.048480 -7.662230 13.281060  0.044220 -0.859590 10.849170  3.080010 -1.259130 13.093070  1.960140 -6.385020 15.493370  0.039760 -6.027580 15.926010  6.397670 -5.446360 15.904470  2.212440 -3.826320 15.493370  0.038910 -7.238500 13.656720  6.047380 -4.497110 14.788910  4.229900 -8.940470 15.494370  0.036990 -6.620310 14.491940  3.800490 -1.422150 13.824490  4.323800 -7.666170 17.709470  0.027430 -4.670140 17.361140  4.668140 -7.995920 15.300010  1.818440 -5.106670 17.710070  0.039120 -3.558320 17.076160  7.221470 -3.056660 15.786700  2.848690 -10.213990 17.709870  0.028060 -8.384040 15.846320  6.269750 -9.586970 17.827510  5.289330 -8.939080 19.916610  0.028910 -7.412600 16.583120  4.004300 -9.236230 17.628890  2.403090 -6.384180 19.921190  0.036780 -4.553130 19.688670  6.813930 -6.950430 18.736620  2.207580 -11.496140 19.921590  0.044220 -7.202440 18.237440  6.204060 -4.652170 18.279890  2.274290 -3.751600 14.174990  7.055620 -4.431590 15.326950  6.839970 -1.615490 12.858380  6.287650 -2.906780 13.362640  6.211320 -0.035980 11.057980  7.199670 -0.695890 12.186310  6.715270  0.458630  8.570790  6.287430  0.118630  9.813900  7.038840  0.128780  6.210620  5.416850  0.146080  7.391040  6.223920  0.046440  3.540910  5.710960 -0.150760  4.794550  5.663420
```
The coordinates of atoms of one frame must be put in a single line. The number of coordinates in one line should be an integral multiple of 3.

#### `force.raw` stores all the forces acting one each atom:
```bash
 1.578933  1.234426 -0.215541 -0.071002  0.040477  0.067627  0.643874  0.817782  1.804038  0.140418 -0.602209 -0.370993 -0.314510 -0.026842  0.110945 -1.154725  0.746525  0.061570  0.393245  0.405126 -0.445063 -0.546614 -0.273987 -1.365228 -0.566814 -0.021014 -0.522211 -0.414802 -0.242908 -0.201699 -0.459542  0.251030  1.150876  1.202525  0.732509  0.011343  0.379316 -0.138404  0.048728  0.034608  0.015164 -0.299721  1.103928 -0.432312  0.801473  1.601354 -0.239050 -0.940189 -0.540637  0.017116  0.299377  0.260585  0.707602  0.428521 -0.315947 -1.127156 -0.358458  0.425075 -0.857528 -0.386716  0.349725  0.288721  1.581775 -1.103007 -0.194441  0.916294 -0.158656 -0.179477 -0.187845 -0.928670 -0.368195 -0.928392  0.490933 -0.205810 -0.385709  0.481216  0.184793 -0.151729  0.092444 -0.272774  1.704720 -1.663963 -0.161329 -0.126070 -0.116684 -0.755480  0.146262  0.810661 -0.273865 -0.229822 -1.939151 -0.414491 -0.357924 -0.980662  0.231375  0.181797  0.773947 -0.348447  0.497017  1.003129  1.214704 -0.319780  0.205269 -0.038593  0.012205 -0.036654 -0.091480  0.297419  5.058554 -1.428782  0.775975  2.200140 -1.459068 -0.361377 -2.407013 -4.248050 -0.665428 -0.575754  0.853238  1.125683  2.065092 -0.547921  0.604318  0.083711  1.084524 -2.235678 -1.765208 -2.241402 -0.987774  0.358767  1.280954  0.795373 -1.306879  3.183898  1.687930  1.458121  1.197451 -0.389000  2.504864 -3.851273 -1.450699 -0.239958  2.453266  0.149779  3.342770  1.194183 -0.583308 -4.377134 -0.359696 -0.217666 -0.603367  3.708960  0.266577 -2.502022  3.753318  1.391766 -1.291656  0.534799  1.061597 -1.012516 -2.187646  0.958831  3.057021  3.344461 -2.691689  0.885689 -1.152736 -1.099464 -1.586589 -3.697602 -0.149795 -3.991565 -1.113915 -0.300073 
-0.796242 -0.523549  0.185499 -0.062476  0.019478  0.585654 -0.591477  0.782278 -0.723076  0.183543 -0.499285 -0.767653 -0.555822 -0.230631 -0.231444  1.854631 -0.441515  0.069979 -0.588252  0.556004  0.476570  0.469344 -0.162928 -0.066125  0.198139 -0.048659 -0.572105  0.097700  0.009616  0.517998  0.009922 -0.051887  1.559134 -0.557552  0.454264 -1.518897 -0.082722 -0.034253  0.498334 -0.085702 -0.279068 -0.059516 -0.981297 -0.981662 -1.475982 -0.357170 -0.214046  0.987730  0.626894  0.711129 -0.658134  1.121814  0.019328  0.409214 -0.502892 -1.571833  1.228355  0.718453  0.601519  0.079543  0.866637 -1.768025 -0.105128  0.351306 -0.123173 -2.307382 -0.006363 -0.038363 -0.018239  0.331385  0.175154 -0.016156 -0.106709 -0.010445  0.523260 -0.008946  0.042684  0.581756 -1.779715 -2.622593 -0.294725  0.219354  1.115243 -0.199485  1.988320  2.411973  0.634199  2.206710 -2.132733  1.468926  0.235352  0.412879  0.096827 -1.604230  0.370091  0.337585 -0.394607 -0.005059 -0.781608  0.183999  1.502932 -1.548655 -0.052225  0.037002  0.586766 -0.398427 -0.171683  0.089223 -4.439925  0.074448 -0.151024 -3.185291  3.377960 -0.056827 -0.924372  1.904382 -0.366016  0.027128  0.868506 -0.678699 -1.005852  3.768328  0.612910  0.740465 -5.918768 -1.243535 -1.042473  1.583030  1.580949 -1.443538 -1.733673 -1.333074 -0.135535  0.807772 -0.870710 -0.070968 -2.905022 -2.537167 -2.087069  1.650645 -0.115878 -0.045157  0.935655  1.580355 -5.196712  2.886700  1.324522  8.778878 -4.780138 -0.463419  4.555694  0.856648  0.924607 -1.251237  2.181023 -0.873816  0.899663 -5.153917  1.620108  1.283406 -3.122272 -0.723248  2.007581  3.229544  1.833105  1.958434  0.507524  1.489690 -0.090309  1.400328 -0.771157 -1.484663  0.181305 -0.362305 
-0.817296 -0.652533  0.163111 -0.059358  0.021358  0.590996 -0.345903  0.946611 -0.608899  0.187809 -0.534664 -0.822442 -0.527495 -0.215686 -0.186113  1.866464 -0.505091  0.051047 -0.561904  0.537500  0.343798  0.461420 -0.163774 -0.069875  0.262378 -0.079401 -0.545043  0.095998  0.015035  0.521154 -0.086009 -0.015147  1.492100 -0.512326  0.477535 -1.619429 -0.084064 -0.031708  0.506260 -0.087614 -0.285133 -0.063650 -0.953855 -1.064325 -1.558479 -0.333947 -0.224804  0.952749  0.637595  0.669916 -0.672518  1.135042  0.025966  0.416070 -0.794508 -1.999859  1.338709  0.643183  0.604309  0.090510  0.926629 -1.843785 -0.101314  0.245574 -0.027807 -2.213285 -0.002366 -0.038619 -0.019653  0.327081  0.169383  0.003558 -0.105657 -0.002898  0.533662 -0.003105  0.042039  0.580295 -1.721280 -2.549653 -0.268653  0.176477  1.025700 -0.268717  1.705056  2.430831  0.584684  2.350041 -2.680709  1.431433  0.171413  0.325353  0.119261 -1.563371  0.380252  0.335705 -0.417948  0.004642 -0.760573  0.613002  2.070197 -1.619411 -0.056211  0.037681  0.586662 -0.391148 -0.174358  0.099096 -4.692153 -0.118222 -0.250013 -4.212673  3.728529  0.178739  0.385628  2.236060 -0.047788 -0.319671  2.177878 -1.249592 -2.265353  4.918514  0.670961  0.348248 -5.486467 -1.039963 -1.042491  1.937924  1.086461 -1.938951 -1.102159 -1.398615  0.673948  0.055069 -0.933922 -0.646088 -2.427474 -2.150145 -0.868974  2.108375 -0.530599  1.369426 -0.612049  1.934326 -3.801038  1.547223  1.209834  6.584486 -3.840466 -0.433028  4.842382  0.371692  1.116928 -0.244437  1.303990 -0.598807  1.177354 -4.024399  1.807698  2.188856 -3.463539 -0.580440  0.470605  1.966497  1.582409  0.240570  0.623809  1.842315  0.822148  1.654484 -0.756428 -1.423327 -0.356307 -0.839817 
 0.152379  0.613277 -0.058308 -0.897083  0.856596  0.105503  0.334027 -1.334169  1.272313 -0.305254 -0.016641  0.040809 -0.190677  1.866219  0.452211  1.577329  0.555882  3.786668  0.976581 -0.143425 -1.733374 -0.625626 -0.102191 -0.725882 -0.217117 -0.169766 -0.033862  0.077544 -0.079065  0.340263  0.548624 -0.665756 -0.393646  1.528669  0.590361  0.378691 -0.088882 -0.092194  0.146344  0.455685 -0.175066 -0.175418 -0.437806  0.271951  0.296939 -0.088569 -0.470948 -1.743660 -0.495757  0.268458  0.143309 -1.527926  1.233384  1.980152  0.029917  0.016319  0.394716 -0.066215  0.153766 -0.304629  0.087430  1.010728  1.948993 -0.134672 -0.103083  0.357260 -0.944634  0.487211 -0.978761 -0.556438 -0.208427 -0.572614 -0.030393  0.031472  0.432354  0.310598 -0.069330 -0.505895  0.221800 -0.069324 -0.248828 -0.001616  0.137280  0.546532  0.455726 -1.291152 -0.230206  0.424279  0.198545 -0.276129 -0.218396  0.095543  0.162350  0.090377  0.084460  0.529220  0.189629  0.364027 -0.472699  0.678549  0.346571 -0.457276 -1.227344 -0.321052  0.384644 -1.280193 -2.023672  0.137649 -0.009053 -0.002567  0.735383  0.233382  0.303002  0.256989  1.041804  2.081307  0.384627 -0.170678  0.044708  0.321619 -0.321648 -0.247037 -0.354585 -2.675377 -0.115219 -0.562595  0.011935 -0.246157  0.284450 -0.151777  1.710730 -2.048125  2.762064 -0.429745  0.015279  0.000372  0.078057  0.370645  0.154256 -0.098233 -0.467515 -0.212572  0.297391  0.210511 -0.112933  0.056877  0.366115 -0.279290 -0.258712 -0.273469 -0.040857 -0.236936 -0.324059  0.012884  0.061984  0.668089 -0.681765 -1.016948 -1.654924  0.533397  0.043479  0.042771 -0.120976 -0.158869  0.196715 -1.020499 -0.393880 -0.060540 -1.284750 -0.350858 -0.334624 -0.183288 -0.299278 -1.021460  1.479499 -0.492139  0.217691 -0.087148 -0.728417  0.088284 -0.132695  0.010846  0.261590  0.806519  0.900883 -0.500462 -0.282761 -0.076658  0.065013 -0.177393 -0.353062 -0.133133  1.034123  0.819740  0.242660  0.271950  0.645691 -0.732696 -0.213595 -0.163600 -0.069917  1.189144 -0.301030 -0.355754  1.264173  0.039560  1.255962  0.005125 -0.037874  0.480094 -0.846259  0.112613 -0.386861  0.037387 -0.804543  1.171021  0.702172 -0.311162 -0.296819 -0.919822 -1.998962  0.550415  0.680693  0.082859 -1.359929 -0.524721 -0.463025 -0.381010  0.061446  0.460643  0.027697  0.768915  0.577939 -1.081449 -1.666534 -1.170111 -2.583686  0.078198 -0.141880  0.658917  0.006952  0.283289 -0.414111 -0.343341  1.106108  0.707474  0.490315  0.051820 -1.398858  4.174293 -2.809781  0.434863 -2.846698  1.269675 -0.093278  1.113734 -1.170431 -0.893177 -2.134888  0.979196  1.176145 -2.866124 -4.408338  1.408871 -1.002744  2.428141  1.073363  2.314712 -1.319112 -0.227067  1.413341  3.614960 -0.107487 -1.032920  3.096624  0.452622  0.882969 -2.389687 -0.867614 
 0.148434  0.560317 -0.002643 -0.962767  0.901991  0.217425  0.253034 -1.276108  1.197737 -0.317957 -0.033304  0.018652 -0.203359  1.906536  0.542085  1.629674  0.617806  3.782880  1.028273 -0.203842 -1.798066 -0.661463 -0.049313 -0.716495 -0.218329 -0.110371 -0.070843  0.078392 -0.076469  0.340712  0.504924 -0.685201 -0.368023  1.457771  0.635587  0.387506 -0.095616 -0.077867  0.143608  0.465730 -0.168294 -0.148920 -0.446678  0.284188  0.324667 -0.091093 -0.531053 -1.956608 -0.416204  0.113098  0.084601 -1.460945  1.189382  2.157036  0.038805  0.023725  0.379155 -0.107769  0.183357 -0.259092  0.084841  1.103940  2.155343 -0.133863 -0.095671  0.355996 -0.943658  0.490075 -0.973163 -0.608664 -0.273022 -0.637340 -0.045199  0.040454  0.406581  0.312693 -0.084242 -0.571660  0.265827 -0.151986 -0.340197  0.002765  0.142045  0.545663  0.389979 -1.248499  0.018609  0.402847  0.229377 -0.283161 -0.204401  0.093912  0.166354  0.090181  0.143224  0.653965  0.171843  0.354307 -0.431491  0.638721  0.326090 -0.393412 -1.194811 -0.313412  0.413993 -1.283902 -2.167510 -0.063521 -0.007779 -0.002719  0.734992  0.232842  0.317139  0.069125  1.155151  2.179063  0.643258 -0.174513  0.047693  0.330801 -0.308224 -0.222389 -0.411132 -2.361148 -0.145916 -0.571141  0.015015 -0.239913  0.284675 -0.236238  1.627564 -1.751486  2.461089 -0.430147  0.018588  0.003590  0.077950  0.384099  0.169668 -0.140922 -0.460442 -0.163754  0.320132  0.200693 -0.111988  0.052243  0.367971 -0.322245 -0.270692 -0.242254 -0.089797 -0.318377 -0.363851  0.011561  0.061780  0.666189 -0.706631 -1.074586 -1.719392  0.533679 -0.004045  0.000007 -0.113715 -0.149957  0.229229 -1.150642 -0.529354  0.027852 -1.373712 -0.349133 -0.382305 -0.187139 -0.305412 -1.083712  1.593133 -0.255165  0.250448 -0.083714 -0.718757  0.098109 -0.120454  0.009011  0.316124  0.808920  0.930697 -0.514884 -0.400588 -0.063353  0.084756 -0.169907 -0.361978 -0.138097  1.177720  0.897157  0.225010  0.329784  0.705031 -0.868253 -0.197963 -0.159943 -0.031489  1.199750 -0.314343 -0.327856  1.383445  0.145203  1.321161  0.006057 -0.040387  0.493440 -0.682947  0.105347 -0.367825  0.022549 -0.847727  1.204964  0.713054 -0.300531 -0.307642 -0.962138 -1.998432  0.438472  0.622121  0.025868 -1.414106 -0.495825 -0.437166 -0.331091  0.073267  0.478818 -0.054649  0.826504  0.456991 -1.134286 -1.705667 -1.158421 -2.652069  0.104426 -0.185475  0.734322 -0.021889  0.181019 -0.485735 -0.471106  0.761333  0.799455 -0.292442  0.322586 -1.414326  3.276247 -2.058144  0.515312 -1.256701  0.674120 -0.232270  1.198644 -0.886044 -1.129732 -1.862400  0.056626  1.464781 -3.444370 -6.911232  0.577585 -0.897681  5.285204  1.742128  3.051212 -0.148005 -0.180469  0.935678  3.150278 -0.457562 -1.421818  4.025359  0.692159  1.318267 -3.645178 -1.112869 
 0.604129 -0.446541 -1.510948 -0.255516  2.185871 -1.438261 -1.089363 -0.502001  0.017567 -0.050338  0.102618 -0.046915  0.240426 -0.241478  1.638313  0.169258  1.056953  0.138703  0.063813 -0.053133  0.484351 -0.285654 -0.561936 -0.500775 -0.536307 -0.004420  0.462283  0.004941 -0.159551  0.129301 -0.119408 -0.498052 -0.157996 -0.437948  0.928794  1.482709 -0.204415 -0.889466 -1.355421  1.327296  1.622196 -0.493756  0.319188  0.083339 -0.072884 -0.108188  0.653705 -0.736878  0.226267 -2.814005 -1.066833  0.680590 -0.413807  1.391451  0.043025  0.104937  0.489517 -1.092236 -2.045251  0.330711 -1.689034  0.354184  0.052432 -0.401946  0.084886 -0.086006  1.688789  0.528252 -0.161052  0.837081 -0.849152 -0.647552  0.601468  0.272110 -0.465326  0.026296 -0.011105 -0.759388 -0.390289  0.473680  2.217674  0.114215 -0.146571  0.150502  0.714382  1.075012 -0.940209  1.178650 -0.608960 -0.577866 -0.619387 -0.795773 -0.969086  0.806992 -0.952452  0.804657  0.014734 -0.215467  0.134403  0.015499  0.123398  0.363599  0.196808 -0.022194 -0.657214 -0.532535 -1.156623 -1.087497  0.261708 -0.023998  0.250519 -0.021728  0.319106 -0.163740  1.637334 -0.059672  0.967353 -0.008216  0.046431  0.499771  0.159876  0.319661  0.122850 -1.373661  0.617772 -1.233912  0.231958  0.540885 -1.142038 -0.175369  0.260146  1.406688 -1.080864  1.018100 -0.551300 -0.391932  0.232666 -1.028707 -0.410523 -0.023198 -0.514709  0.665723  0.162449  1.784089 -0.470173  0.088776 -0.303003 -0.293751 -0.121912 -0.256853  0.828042  0.382136  1.508288 -0.249278  0.531968 -1.732841  0.777462  1.454174  1.433103 -1.265958 -0.477175  1.384789  0.214531 -0.052006 -0.264750 -1.433891  0.630055 -0.930988 -0.028915  0.227892 -0.132433 -0.038210  0.021518  0.585136  0.533226  0.520241 -0.367643  0.908587 -0.654266  1.377168 -2.024649  0.491440 -3.149453 -1.314839 -1.765886 -0.386164  0.055244  0.792766 -0.249723 -0.031570 -0.103004  0.443306  0.946256 -0.874651 -0.078292  1.571202  0.203816  3.845335 -0.080324 -0.218581 -0.160027 -0.233690 -0.090114 -1.159172  1.069336  0.269720 -0.874665 -0.179478 -0.097465  0.543894 -0.436304  0.446623 -0.461944 -2.004718 -1.431187  1.273544  0.414113  0.090644 -0.515694 -0.461012  1.103661  0.099350  0.163302 -0.579699 -0.180375  0.139206 -0.014869  0.220322  0.373043 -0.665017 -1.134409 -0.240903 -0.669387 -0.515301  0.028914  0.015201  0.536385 -0.274368 -0.683104  0.507355  0.730804 -0.080083  0.094654  1.306294  0.823134 -3.614189  2.390342 -2.110590  1.592846 -6.552589  3.671409 -1.376002 -0.369516  0.514739  1.217670 -1.539311  0.524965 -1.409590  6.522577 -4.989664  3.275281  1.228616  9.317260  3.428469  1.016093 -2.258229 -2.242847 -2.182574  0.118627  3.094190 -1.634932 -8.286061 -2.385627 -0.394981  0.902253  0.417109  0.922469  3.393731  0.073624 
```
The format of `force.raw` is the same as `coord.raw`.

#### `energy.raw` contains energy of each frame:
```bash
-321.48014356
-320.22585212
-320.18770483
-379.73225706
-379.13588315
-374.89802095
```

## How to run the training
### STEP 1: Make C executable for data pre-processing
```bash
cd ./Torch-NNMD/c
#(Adapt the makefile for your computer)
make
```

### STEP 2: Pre-process input data
```bash
cd ../test_2 #The input data for test is under this directory
../c/a.out > log
#Modify the ALL_PARAMS.json for your dataset
```
The C code will convert the raw data into symmetry coordinates needed by training and some bin files will be stored in the `./` directory:
```bash
ALL_PARAMS.json,COORD.BIN, ENERGY.BIN, FORCE.BIN, N_ATOMS.BIN, SYM_COORD.BIN, TYPE.BIN, ...
```
The file `all_frame_info.bin.temp` could be deleted and should not affect the training process in the current version.

### STEP 3: Run python script to train
```bash
python3 ../python/train.py
```
## Parameters in ALL_PARAMS.json
**In the current version the read_parameters() function has not been fully completed. All the parameters involved in the data pre-processing procedure need to be modified through the source code. Remember to rebuild the C code after modifying a .c file**
- `cutoff_1`, `cutoff_2`, `cutoff_3`, `cutoff_max`
  - For [DeePMD](https://github.com/deepmodeling/deepmd-kit)-type symmetry coordinates, `cutoff_1` and `cutoff_2` (or `cutoff_max`) correspond to rcs and rc in [its paper](https://arxiv.org/abs/1805.09003).
  - Editing the **read_parameters.c** to change their values.
- `filter_neuron`, `fitting_neuron`, `axis_neuron`
  - `filter_neuron` and `fitting_neuron` are two arrays describing the filter network and fitting network in [DeePMD's paper](https://arxiv.org/abs/1805.09003)
  - `axis_neuron` is the number of columns of the G^(i2) matrix in DeePMD's paper.
- `start_pref_e`, `limit_pref_e`, `start_pref_f`, `limit_pref_f`
  - The total loss is calculated by: `loss_tot = pref_e * loss_e + pref_f * loss_f`. These four parameters determines the amount of contribution of energy `E` and force `F` to the total loss function `loss_tot`
