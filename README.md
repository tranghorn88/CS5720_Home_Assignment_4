# CS5720_Home_Assignment_4
## 1. GAN Architecture
### Question: 
Explain the adversarial process in GAN training. What are the goals of the generator and discriminator, and how do they improve through competition? Diagram of the GAN architecture showing the data flow and objectives of each component.
### Answer:
GAN consists of two networks, including the generator and discriminator. They are trained in an adversarial process. The generator’s goal is to produce fake data, for example, images that look as real as possible, aiming to fool the discriminator into believing that the generated data is real. Meanwhile, the discriminator's role is to distinguish between real data from the training set and fake data produced by the generator. During the training, the generator improves by learning from the discriminators' feedback to create more convincing outputs. At the same time, the discriminators become better at identifying fake data. This competition continues where the generators aim to minimize the discriminator’s ability to detect fakes, meanwhile, the discriminator aims to maximize the classification accuracy until both networks are so good that neither can outperform the other.
#### Diagram of the GAN architecture
![image](https://github.com/user-attachments/assets/519308e1-668d-4a3c-a871-b8df2d4d55af)

## 2. Ethics and AI Harm
### Question:
Choose one of the following real-world AI harms discussed in Chapter 12:
- Representational harm
- Allocational harm
- Misinformation in generative AI
Describe a real or hypothetical application where this harm may occur. Then, suggest two harm mitigation strategies that could reduce its impact based on the lecture.
### Answer:
#### Allocational harm
Company ABC utilizes a machine learning model to automate the screening of job applications. The training data was sourced from past hiring records where historical hiring favored candidates from certain universities, geographic regions, and demographics, e.g., mostly male or white applicants. Harms are applicants from underrepresented backgrounds, such as women, Black or Hispanic candidates, or graduates from less well-known universities, who are qualified for the job, will be filtered out or scored lower in the system and not receive interview opportunities. This AI system reinforces bias in employment practice.

Mitigating strategy can start with fine-tuning and balanced datasets. The hiring model will need to be fine-tuned by collecting and using a smaller, curated dataset that is representative of diverse backgrounds, including races, gender, education, and geography. Doing so will help correct the model's biased pattern inherited from skewed training data.

Another mitigation strategy can be using a fairness auditing tool such as Aequitas to assess model performance across different subgroups, such as gender, race, before using the model. Tools can be used to generate a report, making bias visible and measurable. In the case of this hiring AI system, the tool can reveal that the model disproportionately rejects qualified applicants from underrepresented backgrounds and prompt remediation for the issue. 

## 3.  Programming Task (Basic GAN Implementation)
### Description:
This code implements a GAN using PyTorch to generate handwritten digits similar to those in the MNIST dataset. The GAN consists of two main neural networks:
- A generator that takes random noise as input and generates synthetic images of handwritten digits.
- A discriminator that tries to distinguish between real images (from MNIST) and fake images (from the Generator).
The two networks are trained in an adversarial manner, where the Generator tries to fool the Discriminator, and the Discriminator tries to correctly identify real vs. fake images.
#### Output files:
- samples/fake_images_epoch_0.png contains initial results from the untrained Generator.
- samples/fake_images_epoch_50.png contains intermediate results showing improved digit structure.
- samples/fake_images_epoch_100.png contains the final output showing clearly recognizable digits.
- samples/loss_plot.png contains a line plot comparing Generator and Discriminator losses across epochs.
### Discussion:
This GAN implementation was successfully trained to generate handwritten digits using the MNIST dataset. The progression of generated images and loss metrics illustrates the learning dynamics between the Generator and Discriminator over 100 epochs, specifically:
- Epoch 0: The Generator produces indistinct and noisy blobs, which is expected since the model weights are still untrained. Most outputs are far from resembling digits.
- Epoch 50: The digits become much more defined, demonstrating that the generator has learned significant features from the MNIST dataset, though some samples still show blurring or deformation.
- Epoch 100: The quality continues to improve, with most digits being well-formed and sharp, which demonstrates the generator's improved ability to fool the Discriminator.
  
From the loss plot:
- Generator Loss steadily increased, indicating the model learned to produce more realistic images and forced the Discriminator to improve.
- Discriminator Loss gradually decreased, indicating better classification of real vs. fake digits.
## 4.Programming Task (Data Poisoning Simulation)
### Description:
This code simulates a data poisoning attack on a basic sentiment classifier trained on short movie reviews. The attack targets reviews that reference a specific entity, in this case, "UCM" by flipping their sentiment labels, thus corrupting the training data. This demonstrates how a small, targeted manipulation can severely degrade the model’s performance. 
### Discussion:
The accuracy dropped from 33% to 0% due to poisoning and all test samples were misclassified after the attack.

This simulation clearly demonstrates the devastating impact of a targeted data poisoning attack. Although the poisoned dataset contained only a small number of flipped samples, the effect on the model was severe. The original model, while not perfect, could still identify some positive sentiments. Meanwhile, the poisoned model, however, failed entirely. It predicted all test samples as class 0, regardless of their actual labels. 
## 5. Legal and Ethical Implications of GenAI
### Question:
Discuss the legal and ethical concerns of AI-generated content based on the examples of:
- Memorizing private data (e.g., names in GPT-2)
- Generating copyrighted material (e.g., Harry Potter text)
Do you believe generative AI models should be restricted from certain data during training? Justify your answer.
### Answer:
The deployment of AI raises legal and ethical concerns, especially around violating data privacy and intellectual property rights. In the case of GPT-2, which memorizes private personal data, it shows a critical privacy vulnerability in the AI model. When language models are trained on vast datasets scraped from the internet, they can unintentionally memorize and later reproduce sensitive personal data such as names, email addresses, and contact details, and private conversations that are accidentally made public. This raise a concerns that AI model violate the privacy laws like GDPR in Europe and states privacy statutes in the U.S. Users or individuals who has their data memorized and reproduce can have solid grounds for legal actions since individuals never agrees to have their personal information become part of AI training dataset or have it potentially exposed, reproduced in response to querries.

Another concern is AI generating copyrighted or trademark content, especially when the model is trained on proprietary datasets such as books, movies, etc. These models can generate content that is substantial portions of copyrighted text, mimics distinctive writing styles or character voices without authorization. This potentially violates copyright law, which protects the expression of creative works and their derivative creations. Rights holders can argue that AI companies are engaging in unauthorized reproduction and distribution of their intellectual property, which potentially undermines the monetary value of their creative works. There is also an ethical concern questioning the fairness to the creator whose work is being used to train AI systems that then produce or generate works that potentially compete or substitute for their creations.

Therefore, I believe that AI models should be restricted from certain data. Personal data could be excluded from the training data unless individuals provide explicit, informed consent. Copyrighted materials used for training should require licensing agreements with the rights holders. The fair use doctrine shouldn’t automatically exempt AI systems, especially when the model can produce content that competes with the rights holders. 

Also, restricting access to harmful content can reduce the risk of the AI models reproducing or amplifying such content. However, the goal should be creating sustainable and balanced practice that respects the holder's rights while enabling the technology evolution. This can be improving data curation processes and more advanced privacy-protected training methods.
## 6. Bias & Fairness Tools
### Question:
Visit Aequitas Bias Audit Tool.
Choose a bias metric (e.g., false negative rate parity) and describe:
- What the metric measures
- Why it's important
- How a model might fail this metric
### Answer:
For this question, I am going to analyze the False Negative Rate Parity metric:

False Negative Rate Parity considers an attribute to have parity if every group has the same False Negative Error Rate. It represents the proportion of actual positive cases that the model incorrectly classified as negative within each demographic group. For example, in a loan approval system, true positives mean creditworthy applicants are correctly approved. Meanwhile, false negatives mean creditworthy applicants are incorrectly denied. False Negative Rate refers to the percentage of creditworthy applicants from each group who are wrongly denied loans.

This metric is important in cases where intervention is assistive, such as providing helpful social services, and missing an individual may cause adverse outcomes for them. Using this metric ensures that you are not disproportionately missing people from certain groups. For example, ensuring disease detection rates are equal across racial groups, qualifying people for benefits or support programs equitably, etc.

A model fails False Negative Rate Parity when different demographic groups have significantly different rates of being incorrectly classified as negative when they should be positive. For example, with a cancer screening algorithm, 15% of actual cancer cases were missed in Black patients, while 8% were missed in White patients, causing delayed treatment and worse outcomes for one specific group.









