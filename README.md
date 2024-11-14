# Dynamic Impression Formation using Natural Language

Traditional studies of impression formation generally use paradigms that deviate from real-world impression formation in two critical ways: 1) studies use static photos rather than dynamic stimuli (i.e. videos), and 2) rely on numerical ratings of experimenter-determined constructs of interest (i.e. "Rate this person on trustworthiness", 1-7). Importantly, such work also does not explicitly study the timeseries of impression formation. 
For example, there is not a good account of how expressions of happiness and sadness rapidly in a short amount of time can result in the emergent characterization of such a person as neurotic or unstable. 

In the present work, we develop a novel, data-driven paradigm that uses natural language responses elicited by dynamic, evocative videos to study impression formation. Using advances in natural language processing, we study the timeseries of impression formation in an unconstrained, bottom-up fashion. 

Specifically, I was inspired by memory research where memory recall was modeled as [trajectories through word embedding space](https://www.nature.com/articles/s41562-021-01051-6).

Our paradigm is simple. Participants watch videos, providing natural langauge descriptions of the target whenever they want, as many as they want. They then rate the target on scales from 1-7 on 7 different trait terms derived from prior work. Finally, they provide 5 words as a final, summary impression of the target. 

Our analysis pipeline is inspired by [BERTopic](https://maartengr.github.io/BERTopic/index.html). Rather than using BERT embeddings, we are using FastText embeddings. These embeddings are reduced, then clustered to identify general topics in the free responses. 

These topics characterize a topic space, through which we can model participant impression trajectories. 



 
