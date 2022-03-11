### Parameters

* Map Augmentator
  ```
  ./pathgan/data> python map_augmentator.py -h
  usage: top [-h] [--load_dir LOAD_DIR] [--save_dir SAVE_DIR] [--height_shift HEIGHT_SHIFT] [--width_shift WIDTH_SHIFT]
             [--shift_step SHIFT_STEP] [--rot_prob ROT_PROB] [--n_maps N_MAPS]

  Run Map Augmentator

  optional arguments:
    -h, --help            show this help message and exit
    --load_dir LOAD_DIR   Load directory (default: "dataset/init_maps")
    --save_dir SAVE_DIR   Save directory (default: "dataset/maps")
    --height_shift HEIGHT_SHIFT
                          Number of pixels in which map can be shifted "Up" and "Down" (default: 2)
    --width_shift WIDTH_SHIFT
                          Number of pixels in which map can be shifted "Left" and "Right" (default: 2)
    --shift_step SHIFT_STEP
                          Step in pixels by which map can be shifted (default: 1)
    --rot_prob ROT_PROB   Probability of map to be rotated by "pi/2" (default: 0.5)
    --n_maps N_MAPS       Number of maps to be generated per one map (default: 10)
  ```
* Taks Generator
  ```
  ./pathgan/data> python task_generator.py -h
  usage: top [-h] [--load_dir LOAD_DIR] [--save_dir SAVE_DIR] [--min_length MIN_LENGTH] [--n_tasks N_TASKS]

  Run Task Generator

  optional arguments:
    -h, --help            show this help message and exit
    --load_dir LOAD_DIR   Load directory (default: "dataset/maps")
    --save_dir SAVE_DIR   Save directory (default: "dataset/tasks")
    --min_length MIN_LENGTH
                          Minimal Euclidian distance between "start" and "goal" points (default: 30)
    --n_tasks N_TASKS     Number of tasks to be generated per one map (default: 100)
  ```
* ROI Generator
  ```
  pathgan\data> python roi_generator.py -h
  usage: top [-h] [--start START] [--to TO] [--map_dir MAP_DIR] [--task_dir TASK_DIR] [--save_dir SAVE_DIR]
             [--n_runs N_RUNS]

  Run ROI Generator

  optional arguments:
    -h, --help           show this help message and exit
    --start START        Map "from" which ROI will be obtained (default: 0)
    --to TO              Map "to" which ROI will be obtained (default: None)
    --map_dir MAP_DIR    Maps directory (default: "dataset/maps")
    --task_dir TASK_DIR  Tasks directory (default: "dataset/tasks")
    --save_dir SAVE_DIR  Save directory (default: "dataset/tasks")
    --n_runs N_RUNS      Number of times searching algorithm will be runned per one map (default: 50)
  ```
