from langchain_ollama import OllamaEmbeddings
from langchain.evaluation import load_evaluator
from scipy.spatial.distance import cosine

def main():
    # Get embedding for a word.
    embedding_function = OllamaEmbeddings(model="llama3.1")
    vector1 = embedding_function.embed_query("apple")
    vector2 = embedding_function.embed_query("iphone")
    print(f"Vector for 'apple': {vector1}")
    print(f"Vector for 'iphone': {vector2}")

    # Calculate cosine similarity
    distance = cosine(vector1, vector2)
    print(f"Cosine distance between 'apple' and 'iphone': {distance}")

    # Compare vector of two words using evaluator
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

if __name__ == "__main__":
    main()