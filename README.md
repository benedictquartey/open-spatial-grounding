# open-spatial-grounding
Out of the box open source package for grounding open vocabulary objects with spatial constraints in 3D maps.

![Splash](assets/images/splash.gif?raw=true)

## Installation
* Create conda env and install relevant packages: ```conda env create -f environment.yml```
* Activate your conda environment: ```conda activate osg```
* Install either [Mobile SAM](https://github.com/ChaoningZhang/MobileSAM.git) or [Segment Anything](https://github.com/facebookresearch/segment-anything.git)
  * Mobile Sam: ```pip install git+https://github.com/ChaoningZhang/MobileSAM.git```
  * Segment Anything: ```pip install git+https://github.com/facebookresearch/segment-anything.git```
* Get SAM or Mobile Sam model checkpoints and place it in the [model_ckpts](osg/model_ckpts/) folder

## Running Instructions
* Grab sample data from [drive](https://)
* Walkthrough the [demo notebook](demo_notebook.ipynb)
* Run the [demo_visualize_results.py](demo_visualize_results.py) to visualize results
  * To visualize grounded detections after spatial grounding run :\
 ```python demo_visualize_results.py --result_dir results --show_elements --remove_roof --show_origin```
  * To visualize Referent Semantic Map run:\
```python demo_visualize_results.py --result_dir results --rsm --remove_roof```

## Citation

The methods implemented in this codebase were proposed in the paper["Verifiably Following Complex Robot Instructions with Foundation Models"](https://arxiv.org/pdf/2402.11498). If you find any part of this code useful, please consider citing:

```bibtex
@article{quartey2024verifiably,
  title={Verifiably Following Complex Robot Instructions with Foundation Models},
  author={Quartey, Benedict and Rosen, Eric and Tellex, Stefanie and Konidaris, George},
  journal={arXiv preprint arXiv:2402.11498},
  year={2024}
}
```

This codebase leverages a number of incredible works from the research community. Do consider consider citing the original authors as well, depending on the modules you use.
<details>
<summary>List of citations</summary>

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
</details> 
