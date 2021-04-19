import os
import argparse

from utils import MapAugmentator 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Run Map Augmentator')

    parser.add_argument('--load_dir', default='dataset/init_maps',
                        help='Load directory (default: "dataset/init_maps")')
    
    parser.add_argument('--save_dir', default='dataset/maps',
                        help='Save directory (default: "dataset/maps")')
    
    parser.add_argument('--height_shift', type=int, default=2,
                        help='Number of pixels in which map can be shifted "Up" and "Down" (default: 2)')
    
    parser.add_argument('--width_shift', type=int, default=2,
                        help='Number of pixels in which map can be shifted "Left" and "Right" (default: 2)')
    
    parser.add_argument('--shift_step', type=int, default=1,
                        help='Step in pixels by which map can be shifted (default: 1)')
    
    parser.add_argument('--rot_prob', type=float, default=0.5,
                        help='Probability of map to be rotated by "pi/2" (default: 0.5)')
    
    parser.add_argument('--n_maps', type=int, default=10,
                        help='Number of maps to be generated per one map (default: 10)')
    
    args = parser.parse_args()

    print('============== Map Augmentation Started ==============')
    map_names = sorted(os.listdir(args.load_dir))

    map_augmentator = MapAugmentator()
    map_augmentator.set_parameters(height_shift = args.height_shift,
                                   width_shift = args.width_shift,
                                   shift_step = args.shift_step,
                                   rot_prob = args.rot_prob,
                                   n_maps = args.n_maps,
                                   load_dir = args.load_dir,
                                   save_dir = args.save_dir)

    for map_name in map_names:
        done = map_augmentator.augment(map_name=map_name)
        done = map_augmentator.save()
    print('=====================  Finished! =====================')
