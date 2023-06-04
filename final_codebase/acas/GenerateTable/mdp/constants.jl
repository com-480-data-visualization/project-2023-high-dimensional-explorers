export COC,WL,WR,SL,SR, stateType, actType, ACTIONS, discount_f, RANGES,THETAS,PSIS,OWNSPEEDS,INTRPSEEDS, interp, turns

using DelimitedFiles

# ADVISORY INDICES
COC=0
WL=1
WR=2
SL=3
SR=4

# State Type:
stateType = Tuple{Float64,Float64,Float64,Float64,Float64,Int}
actType = Int
ACTIONS = [COC,WL,WR,SL,SR]

# Default parameters
discount_f = 1.0

RANGES = readdlm("ranges.txt", ',')[:]
# THETAS = readdlm("thetas.txt", ',')[:]
# PSIS = readdlm("psis.txt", ',')[:]

n_theta = 21
n_psis = 21

THETAS = Array(LinRange(-pi,pi,n_theta))
PSIS   = Array(LinRange(-pi,pi,n_psis))

OWNSPEEDS = readdlm("ownspeeds.txt", ',')[:]
INTRSPEEDS = readdlm("intrspeeds.txt", ',')[:]

interp = LocalGIFunctionApproximator(RectangleGrid(RANGES,THETAS,PSIS,OWNSPEEDS,INTRSPEEDS,ACTIONS)) # Create the local function approximator using the grid

### Dictionaries to define transitions ###
probs = [0.5,0.25,0.25]
turns = Dict(COC=>([0.34,0.33,0.33],[0.0,1.5,-1.5].* pi/180.0 ),
              WL=>(probs,[1.5,2.0,1.25].* pi/180.0 ),
              WR=>(probs,[-1.5,-1.25,-2.0].* pi/180.0 ),
              SL=>(probs,[3.0,4.0,2.0].* pi/180.0 ),
              SR=>(probs,[-3.0,-2.0,-4.0].* pi/180.0 ),
              -1=>([0.34,0.33,0.33],[0.0,1.5,-1.5].* pi/180.0 )) # FOR v5, 0, 1, -1
