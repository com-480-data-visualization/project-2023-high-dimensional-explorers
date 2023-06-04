#
Adapted from https://github.com/sisl/HorizontalCAS

Let $DVP_HOME denote the top level directory of this repository.
We assume that <$DVP_HOME> is at `<tilde>/data-visualization-project` where `tilde` 
is set in `config.yaml` in the `paths.tilde` member if not equal to the system "~"
(this is necessary for some setups, like Kyle's Idiap desktop, where using the 
actual "~" is not practical).

In <$DVP_HOME>/acas/GenerateTable/ run
`julia install_packages.jl`
`julia -p1 SolveMDP.jl`

This builds the raw value table as an hdf 5 file.

Next run `python3 genTrainingData5d.py`, which reads the output of the file above,
does some scaling and change of variables. And writes an hdf 5 file ready for 
training a DNN.

# Description of the 5d dataset:
r (ft): Distance from ownship to intruder
t (rad): Angle to intruder relative to ownship heading direction
p (rad):  Heading angle of intruder relative to ownship heading direction
vown (ft/s): Speed of ownship
vint (ft/s): Speed of intruder

NB: 1 ft/s is about 1.1 km/hr.

ft_kilometer = 3280.84
seconds_per_hour = 60 * 60
ft_per_second_to_km_per_hour = seconds_per_hour / ft_kilometer

Can alternatively rewrite as

x (ft): X Distance from ownship to intruder
y (ft): Y Distance from ownship to intruder
p (rad):  Heading angle of intruder relative to ownship heading direction
vown (ft/s): Speed of ownship
vint (ft/s): Speed of intruder

https://arxiv.org/pdf/1810.04240.pdf
