[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# MaskedFace-Net dataset, split using methods proposed by the authors of BinaryCoP

This directory contains image samples of the MaskedFace-Net dataset. We split them into a total of four detection classes following methods described by the authors of BinaryCoP: 
(i) correctly masked face, with full coverage of the nose, mouth and chin;
(2) incorrectly masked face with uncovered chin;
(3) incorrectly masked face with uncovered mouth and nose; and
(4) incorrectly masked face with uncovered nose.

# Licenses
<b>In the following the licenses of the original [FFHQ-dataset](https://github.com/NVlabs/ffhq-dataset):</b>
The individual images were published in Flickr by their respective authors under either <a href=https://creativecommons.org/licenses/by/2.0/>Creative Commons BY 2.0</a>, <a href=https://creativecommons.org/licenses/by-nc/2.0/>Creative Commons BY-NC 2.0</a>, <a href=https://creativecommons.org/publicdomain/mark/1.0/>Public Domain Mark 1.0</a>, <a href=https://creativecommons.org/publicdomain/zero/1.0/>Public Domain CC0 1.0</a>, or <a href=http://www.usa.gov/copyright.shtml>U.S. Government Works</a> license. All of these licenses allow free use, redistribution, and adaptation for non-commercial purposes. However, some of them require giving appropriate credit to the original author, as well as indicating any changes that were made to the images. The license and original author of each image are indicated in the metadata.

<ul>
  <li>https://creativecommons.org/licenses/by/2.0/</li>
  <li>https://creativecommons.org/licenses/by-nc/2.0/</li>
  <li>https://creativecommons.org/publicdomain/mark/1.0/</li>
  <li>https://creativecommons.org/publicdomain/zero/1.0/</li>
  <li>http://www.usa.gov/copyright.shtml
</ul>

The dataset itself (including JSON metadata, download script, and documentation) is made available under <a href=https://creativecommons.org/licenses/by-nc-sa/4.0/>Creative Commons BY-NC-SA 4.0</a> license by NVIDIA Corporation. You can use, redistribute, and adapt it for non-commercial purposes, as long as you (a) give appropriate credit by citing our paper, (b) indicate any changes that you've made, and (c) distribute any derivative works under the same license.
https://creativecommons.org/licenses/by-nc-sa/4.0/

<b>The licenses of MaskedFace-Net dataset:</b> The dataset is made available under <a href=https://creativecommons.org/licenses/by-nc-sa/4.0/>Creative Commons BY-NC-SA 4.0</a> license by NVIDIA Corporation. 
You can use, redistribute, and adapt it for non-commercial purposes, as long as you 
<ol type="a">
<li> give appropriate credit by citing our papers: <ol type="1"><li>Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020, <a href=https://doi.org/10.1016/j.smhl.2020.100144>DOI:10.1016/j.smhl.2020.100144</a></li> <li> Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 2020, <a href=https://www.techscience.com/CMES/v124n3/39927>DOI:10.32604/cmes.2020.011663</a></li></ol></li>
<li> indicate any changes that you've made, </li>
<li>and distribute any derivative works under the same license. https://creativecommons.org/licenses/by-nc-sa/4.0/ </li>
</ol>

# Acknowledgements
> Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020, <a href=https://doi.org/10.1016/j.smhl.2020.100144>DOI:10.1016/j.smhl.2020.100144</a> 

```
@Article{cabani.hammoudi.2020.maskedfacenet,
    title={MaskedFace-Net -- A Dataset of Correctly/Incorrectly Masked Face Images in the Context of COVID-19},
    author={Adnane Cabani and Karim Hammoudi and Halim Benhabiles and Mahmoud Melkemi},
    journal={Smart Health},
    year={2020},
    url ={http://www.sciencedirect.com/science/article/pii/S2352648320300362},
    issn={2352-6483},
    doi ={https://doi.org/10.1016/j.smhl.2020.100144}
}
```

>Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 2020, <a href=https://www.techscience.com/CMES/v124n3/39927>DOI:10.32604/cmes.2020.011663</a>

```
@Article{cmes.2020.011663,
    title={Validating the Correct Wearing of Protection Mask by Taking a Selfie: Design of a Mobile Application “CheckYourMask” to Limit the Spread of COVID-19},
    author={Karim Hammoudi, Adnane Cabani, Halim Benhabiles, Mahmoud Melkemi},
    journal={Computer Modeling in Engineering \& Sciences},
    volume={124},
    year={2020},
    number={3},
    pages={1049--1059},
    url={http://www.techscience.com/CMES/v124n3/39927},
    issn={1526-1506},
    doi={10.32604/cmes.2020.011663}
}
```

>Fasfous, Nael, et al., "Binarycop: Binary neural network-based covid-19 face-mask wear and positioning predictor on edge devices", 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). IEEE, 2021.

```
@inproceedings{bcop,
    author={Fasfous, Nael and Vemparala, Manoj-Rohit and Frickenstein, Alexander and Frickenstein, Lukas and Badawy, Mohamed and Stechele, Walter},
    booktitle={2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
    title={BinaryCoP: Binary Neural Network-based COVID-19 Face-Mask Wear and Positioning Predictor on Edge Devices},
    year={2021},
    pages={108-115},
    doi={10.1109/IPDPSW52791.2021.00024}
}
```

