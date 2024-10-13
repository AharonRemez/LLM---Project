# LLM - Project
This project implements a GPT-2 scale model built from scratch, but to save time and resources, pre-trained weights from GPT-2 medium were used. The model was fine-tuned on the [ChatDoctor dataset](https://huggingface.co/datasets/avaliev/chat_doctor) to handle medical question answering tasks.
## Project Overview
#### The goal of this project is to build a conversational agent capable of answering medical-related questions. The model was fine-tuned on medical dialogue data, with a focus on providing medically relevant responses. While the architecture is based on GPT-2, training was limited due to resource constraints.
- #### Training Loss: 2.472
- #### Validation Loss: 2.501
#### The model was trained for approximately 400 steps due to resource limitations, which means it did not achieve optimal convergence. More training could improve performance further.

## Example
- **Question:**
Hi Doctor, I am a male and I have had a stomachache for a few days now because of the pain I have laxatives a lot. What should I do?

- **Response:**
Hi, thanks for asking your query to ChatDoctor. I tested blood sugar with enzymatic tests, high new plasma Ig output, and gastric ultrasound blood pressure. I am very sorry for your complaint because you are suffering from diabetes, which could be due to the condition. I would suggest intravenous folic acid twice daily. You can also lactate rectally every day. I will suggest following an X-ray study if there is no improvement with digestive structure. A consultation with a physician is also recommended. Thank you.

## Model Training Details
- **Environment:** The model was trained in a Colab environment using PyTorch.
- **Device Setup:** Wherever you see #device in the code, it refers to the system's hardware configuration (e.g., CPU or GPU).
- **Pre-trained Weights:** The model uses GPT-2 medium pre-trained weights from Hugging Face as the base for further fine-tuning.
## Dataset
The dataset used for training is the ChatDoctor dataset from Hugging Face, which contains a wide variety of medical-related questions and answers.

- **To download the dataset, you can use the following command in Google Colab:**
```python
!pip install datasets
from datasets import load_dataset
dataset = load_dataset("avaliev/chat_doctor")
```
## Limitations
Due to limited computational resources, the model was trained for only 400 steps, which is insufficient for full convergence. Further training and optimization are recommended for better performance.
