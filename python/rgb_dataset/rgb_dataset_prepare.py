import os
import glob
import shutil

scenes_path = '/media/daniele/data/daniele/datasets/rgbd-dataset/rgbd-scenes'
aligned_path = '/media/daniele/data/daniele/datasets/rgbd-dataset/rgbd-scenes_aligned'

scenes = {
    #'desk': ['desk_3'],
    #'kitchen_small': ['kitchen_small_1']
    'meeting_small': ['meeting_small_1'],
    'table': ['table_1'],
    #'table_small': ['table_small_1', 'table_small_2']
}

for scene_name, subs in scenes.iteritems():
    print "Scene:", scene_name
    for sub_scene in subs:
        scene_path = os.path.join(scenes_path, scene_name, sub_scene)
        files = glob.glob(os.path.join(scene_path, "*.*"))
        print " - Sub Scene:", sub_scene
        for f in files:
            basename = os.path.basename(f)
            if '_depth' not in basename:
                newname = basename.replace(".png", "_rgb.png")
                newpath = os.path.join(scenes_path, scene_name, sub_scene, newname)
                shutil.move(f, newpath)
                print basename, newname
