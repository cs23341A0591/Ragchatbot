"""Generates sample_document.pdf (topic: Machine Learning Basics) for testing the chatbot."""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

sections = [
 ("Introduction to Machine Learning",
  "Machine Learning (ML) is a branch of artificial intelligence in which computers learn patterns from data instead of being explicitly programmed. "
  "The term was popularised by Arthur Samuel in 1959, who built a checkers-playing program that improved with experience. "
  "Today ML powers recommendation systems, spam filters, voice assistants, medical image analysis, fraud detection and self-driving cars."),
 ("Types of Machine Learning",
  "Supervised learning uses labelled data, where every training example has an input and a correct output. Examples include linear regression, logistic regression, decision trees, random forests and support vector machines. "
  "Unsupervised learning works on unlabelled data and finds hidden structure; clustering with K-Means and dimensionality reduction with PCA are common examples. "
  "Reinforcement learning trains an agent to take actions in an environment so as to maximise cumulative reward. It is used in game playing and robotics. "
  "Semi-supervised and self-supervised learning combine small amounts of labelled data with large amounts of unlabelled data."),
 ("The Machine Learning Workflow",
  "A typical ML project has these stages: problem definition, data collection, data cleaning and preprocessing, feature engineering, model selection, training, evaluation and deployment. "
  "Data is usually split into a training set (about 70-80 percent), a validation set and a test set. The test set is used only once, at the end, to estimate real-world performance. "
  "Cross-validation, especially k-fold cross-validation, gives a more reliable estimate when data is limited."),
 ("Overfitting and Underfitting",
  "Overfitting happens when a model memorises the training data, including noise, and performs poorly on new data. Underfitting happens when a model is too simple to capture the real pattern. "
  "Ways to reduce overfitting include collecting more data, using regularisation (L1 and L2), dropout in neural networks, early stopping and reducing model complexity. "
  "The bias-variance trade-off describes the balance: high bias leads to underfitting, high variance leads to overfitting."),
 ("Evaluation Metrics",
  "For classification the common metrics are accuracy, precision, recall, F1-score and ROC-AUC. Precision is the fraction of predicted positives that are correct; recall is the fraction of actual positives that are found. "
  "The F1-score is the harmonic mean of precision and recall. For regression the common metrics are Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R-squared. "
  "A confusion matrix shows true positives, false positives, true negatives and false negatives."),
 ("Neural Networks and Deep Learning",
  "An artificial neural network consists of layers of connected neurons. Each neuron computes a weighted sum of its inputs and applies an activation function such as ReLU, sigmoid or tanh. "
  "Networks are trained with backpropagation and gradient descent, which adjust the weights to minimise a loss function. "
  "Deep learning uses networks with many layers. Convolutional Neural Networks (CNNs) are used for images, Recurrent Neural Networks (RNNs) and LSTMs for sequences, and Transformers for language tasks. "
  "Large Language Models (LLMs) such as GPT and Qwen are Transformer models trained on huge text corpora."),
 ("Retrieval-Augmented Generation (RAG)",
  "RAG combines a retriever with a language model. Documents are split into chunks, converted to embeddings and stored in a vector database such as Chroma or FAISS. "
  "When a user asks a question, the most similar chunks are retrieved and given to the LLM as context, so that the answer is grounded in the documents and hallucination is reduced. "
  "Important RAG parameters are chunk size, chunk overlap, the number of retrieved chunks (k), the embedding model and the LLM temperature."),
 ("Popular Tools and Libraries",
  "Python is the main language for ML. Key libraries are NumPy and Pandas for data handling, Matplotlib and Seaborn for visualisation, Scikit-learn for classical ML, "
  "TensorFlow, Keras and PyTorch for deep learning, and Hugging Face Transformers for pre-trained language models. Google Colab and Jupyter Notebook are popular environments for experimentation."),
 ("Career Paths in Machine Learning",
  "Common roles include Machine Learning Engineer, Data Scientist, Data Analyst, AI Research Scientist and MLOps Engineer. "
  "Useful skills are Python, statistics, linear algebra, SQL, model deployment, and building portfolio projects such as chatbots, recommendation engines and image classifiers."),
]

styles = getSampleStyleSheet()
doc = SimpleDocTemplate("sample_document.pdf", pagesize=A4, title="Machine Learning Basics")
story = [Paragraph("Machine Learning Basics - A Beginner's Guide", styles["Title"]), Spacer(1, 12)]
for h, body in sections:
    story += [Paragraph(h, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 10)]
doc.build(story)
print("Created sample_document.pdf")
