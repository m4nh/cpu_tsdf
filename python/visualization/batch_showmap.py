from subprocess import call


comm = "python showmap.py _map_file:=/media/daniele/data/daniele/datasets/rgb-dataset-masks-squared-maps/scene_{0}.map _roll:=0.0 _pitch:=-0.0 _yaw:=0.0 _min_cluster_size:=50 _min_w:={1} _output_name:=scene_{0}_w{1} _debug:=false"

scenes = [
    "01", "02", "03", "04", "09", "11", "12", "13"
]
weights = [0.01, 0.02, 0.05, 0.1, 0.2]

for s in scenes:
    for w in weights:
        current_command = comm.format(s, w)
        call(current_command.split())
