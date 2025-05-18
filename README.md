# Predicting When and What to Explain from Multimodal Eye Tracking and Task Signals

This repository contains the code for the data preprocessing, machine learning models, and hyperparameter search described in our paper.

## Reference 
You can find the free PDF [here](https://kclpure.kcl.ac.uk/admin/files/272361658/TAC_When_to_Explain_2024.pdf) and similar, related research [here](https://lwachowiak.github.io/project/explainablerobots/). 

```
@ARTICLE{wachowiak_explainFromGaze_2024,
  author={Wachowiak, Lennart and Tisnikar, Peter and Canal, Gerard and Coles, Andrew and Leonetti, Matteo and Celiktutan, Oya},
  journal={IEEE Transactions on Affective Computing}, 
  title={Predicting When and What to Explain From Multimodal Eye Tracking and Task Signals}, 
  year={2025},
  volume={16},
  number={1},
  pages={179-190},
  doi={10.1109/TAFFC.2024.3419696}}
```
## Abstract 
While interest in the field of explainable agents increases, it is still an open problem to incorporate a proactive explanation component into a real-time human–agent collaboration. Thus, when collaborating with a human, we want an agent to identify critical moments requiring explanations. We differentiate between situations requiring explanations about the agent's decision-making and assistive explanations supporting the user. In order to detect these situations, we analyze eye tracking signals of participants placed into a collaborative virtual cooking scenario. Firstly, we show how users' gaze patterns differ between moments of user confusion, the agent making errors, and the user successfully collaborating with the agent. Secondly, we evaluate different state-of-the-art models on the task of predicting whether the user is confused or the agent makes errors using gaze- and task-related data. An ensemble of MiniRocket classifiers performs best, especially when updating its predictions with high frequency based on input samples capturing time windows of 3 to 5 seconds.
We find that gaze is a significant predictor of when and what to explain. Gaze features are crucial to our classifier's accuracy, with task-related features benefiting the classifier to a smaller extent.

## Overcooked Environment Used for Data Collection
https://github.com/lwachowiak/overcooked-demo 
