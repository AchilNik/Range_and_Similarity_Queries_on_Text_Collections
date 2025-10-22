!pip install pandas
!pip install datasketch
import pandas as pd
from datasketch import MinHash, MinHashLSH


def letter_normalization(letter):
    """
    Normalize a letter to a numerical value based on its position in the alphabet.
    'A' -> 1, 'B' -> 2, ..., 'Z' -> 26

    Parameters:
    - letter: A single character string representing the letter to normalize.

    Returns:
    - An integer representing the normalized value of the letter.
    """
    return ord(letter.upper()) - ord('A') + 1


class Node:
    def __init__(self, point, axis=0, left=None, right=None):
        self.point = point  # Tuple: (normalized_surname, awards, publications, original index in dataset)
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None

    def build(self, points, depth=0):
        if not points:
            return None

        # Number of dimensions is 3 (normalized_surname, awards, publications)
        k = 3
        axis = depth % k

        # Sort point list and choose median as pivot element
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2

        # Create node and construct subtrees
        return Node(
            point=points[median],
            axis=axis,
            left=self.build(points[:median], depth + 1),
            right=self.build(points[median+1:], depth + 1),
        )

    def insert(self, point, depth=0, node=None):
        if node is None:
            if self.root is None:
                self.root = Node(point, axis=depth % 3) # 3 dimensions
                return self.root
            else:
                node = self.root

        # Insert recursively based on the current axis
        axis = node.axis
        if point[axis] < node.point[axis]:
            if node.left is None:
                node.left = Node(point, axis=(depth+1) % 3)
            else:
                self.insert(point, depth+1, node.left)
        else:
            if node.right is None:
                node.right = Node(point, axis=(depth+1) % 3)
            else:
                self.insert(point, depth+1, node.right)
        return node

    def query(self, node, min_val, max_val, depth=0, results=None):
        if node is None:
            return results if results else []

        if results is None:
            results = []

        # Check current node against criteria
        if all(min_val[d] <= node.point[d] <= max_val[d] for d in range(3)):
            results.append(node.point)

        axis = depth % 3

        # Check subtrees; consider left subtree if it could contain points within bounds
        if node.left is not None and min_val[axis] <= node.point[axis]:
            self.query(node.left, min_val, max_val, depth+1, results)

        # Consider right subtree if it could contain points within bounds
        if node.right is not None and node.point[axis] <= max_val[axis]:
            self.query(node.right, min_val, max_val, depth+1, results)

        return results



df = pd.read_csv("data.csv")
def preprocess_data_with_indices(df):
    # Include an index or identifier in your points
    points = []
    for index, row in df.iterrows():
        normalized_surname = letter_normalization(row['surname'][0])
        awards = row['awards']
        publications = row['publications']
        points.append((normalized_surname, awards, publications, index))  # Include index
    return points

points = preprocess_data_with_indices(df)
kdtree = KDTree()
kdtree.root = kdtree.build(points)
min_val = [letter_normalization('A'), 4, 100]
max_val = [letter_normalization('G'), float('inf'), 2000]
results = kdtree.query(kdtree.root, min_val, max_val)
matching_indices = [result[-1] for result in results]


# # Extract 'education' descriptions for these indices
education_descriptions = df.loc[matching_indices, 'education'].tolist()



!pip install transformers
!pip install datasketch
!pip install faiss-gpu

from transformers import BertTokenizer, BertModel
import torch

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

embeddings = [np.squeeze(embedding(text)) for text in education_descriptions]

# Squeeze the embeddings to make them 1-dimensional
squeezed_embeddings = [np.squeeze(embedding) for embedding in embeddings]

# Compute similarity scores based on cosine similarity of embeddings
n = len(squeezed_embeddings)
similarity_scores = [[0 for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        similarity_scores[i][j] = 1 - cosine_distance(squeezed_embeddings[i], squeezed_embeddings[j])
        similarity_scores[j][i] = similarity_scores[i][j]


for i in range(n):
    for j in range(i+1, n):
        if similarity_scores[i][j] > 0.5:  # Adjust threshold as needed
            scientist1_name = df.loc[i, 'surname']
            scientist2_name = df.loc[j, 'surname']


import numpy as np
import faiss

def create_lsh_index(embeddings, n_bits=256):
    """
    Create an LSH index for embeddings.

    Args:
    - embeddings: numpy array of shape (n_samples, n_features)
    - n_bits: number of bits for the LSH. More bits mean more buckets and finer granularity.

    Returns:
    - index: The LSH index.
    """
    d = embeddings.shape[1]  # Dimensionality of the embeddings
    index = faiss.IndexLSH(d, n_bits)
    index.add(embeddings)  # Add embeddings to the index
    return index

# Convert list of embeddings to a numpy array
embeddings_array = np.vstack(embeddings)

# Create LSH index
lsh_index = create_lsh_index(embeddings_array, n_bits=256)

# Example query to find similar items
def query_similar_items(query_embedding, index, n=2):
    """
    Query the index for n most similar items to the query_embedding.

    Args:
    - query_embedding: The query embedding vector.
    - index: The LSH index.
    - n: Number of similar items to find.

    Returns:
    - D: distances of the n most similar items.
    - I: indices of the n most similar items.
    """
    D, I = index.search(query_embedding.reshape(1, -1), n)  # Reshape for single query
    return D, I

# Query for similar items
query_embedding = embeddings_array[0]
distances, indices = query_similar_items(query_embedding, lsh_index, n=5)

query_index = matching_indices[0]

print("Pairs of similar scientists based on LSH query:")
for i, index in enumerate(indices[0]):
    if i == 0:
        continue  # Skip the first one if it's the query itself
    similar_scientist_name = df.iloc[matching_indices[index]]['surname']
    print(f"Similar to {df.iloc[matching_indices[query_index]]['surname']}: {similar_scientist_name} with LSH distance: {distances[0][i]:.2f}")



class Point:
    """Represents a point in 2D space."""
    def __init__(self, x, y, data=None):
        self.x = x
        self.y = y
        self.data = data  # Additional data associated with the point

class Rect:
    """Represents a rectangular area in 2D space."""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, point):
        """Check if the rect contains a point."""
        return (self.x <= point.x < self.x + self.width and
                self.y <= point.y < self.y + self.height)

    def intersects(self, range):
        """Check if the rect intersects another rect."""
        return not (range.x > self.x + self.width or
                    range.x + range.width < self.x or
                    range.y > self.y + self.height or
                    range.y + range.height < self.y)

class QuadTree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary  # Rect
        self.capacity = capacity
        self.points = []
        self.divided = False

    def subdivide(self):
        """Divides the current quad into four subquads."""
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.width / 2, self.boundary.height / 2
        self.northeast = QuadTree(Rect(x + w, y, w, h), self.capacity)
        self.northwest = QuadTree(Rect(x, y, w, h), self.capacity)
        self.southeast = QuadTree(Rect(x + w, y + h, w, h), self.capacity)
        self.southwest = QuadTree(Rect(x, y + h, w, h), self.capacity)
        self.divided = True

    def insert(self, point):
        """Inserts a point into the QuadTree."""
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()

        return (self.northeast.insert(point) or
                self.northwest.insert(point) or
                self.southeast.insert(point) or
                self.southwest.insert(point))

    def query(self, range, found_points):
        """Finds the points contained within a range."""
        if not self.boundary.intersects(range):
            return found_points

        for point in self.points:
            if range.contains(point):
                found_points.append(point)

        if self.divided:
            self.northwest.query(range, found_points)
            self.northeast.query(range, found_points)
            self.southwest.query(range, found_points)
            self.southeast.query(range, found_points)

        return found_points

if __name__ == "__main__":
    boundary = Rect(0, 0, 200, 200)
    qt = QuadTree(boundary, 4)

    # Insert some points
    qt.insert(Point(50, 50, "Data1"))
    qt.insert(Point(150, 150, "Data2"))
    qt.insert(Point(25, 25, "Data3"))
    qt.insert(Point(125, 125, "Data4"))

    # Define a query range
    query_range = Rect(0, 0, 100, 100)
    found_points = qt.query(query_range, [])



def letter_normalization(letter):
    return ord(letter.upper()) - ord('A') + 1

# Initialize the QuadTree boundary to cover the expected range of data
boundary = Rect(0, 0, 26, max(df['awards']) + 1)
qt = QuadTree(boundary, capacity=4)

for index, row in df.iterrows():
    x = letter_normalization(row['surname'][0])
    y = row['awards']
    qt.insert(Point(x, y, index))  # Store DataFrame index in Point's data

# Define a query range based on your criteria (e.g., surnames A-G, at least 4 awards)
query_range = Rect(1, 4, 6, max(df['awards']) - 4)
found_points = qt.query(query_range, [])

# Extract indices from found points
matching_indices = [point.data for point in found_points]
education_descriptions = df.loc[matching_indices, 'education'].tolist()


# Initialize tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

embeddings = np.array([embedding(text) for text in education_descriptions])


# Flatten embeddings and compute cosine similarity
embeddings = np.squeeze(embeddings)  # Ensure embeddings are 2D (n_samples, n_features)
n = len(embeddings)
similarity_scores = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        similarity_scores[i][j] = 1 - cosine_distance(embeddings[i], embeddings[j])
        similarity_scores[j][i] = similarity_scores[i][j]

# Identify pairs with high similarity
for i in range(n):
    for j in range(i + 1, n):
        if similarity_scores[i][j] > 0.5:  # Threshold of 0.5 for similarity
            scientist1_name = df.loc[matching_indices[i], 'surname']
            scientist2_name = df.loc[matching_indices[j], 'surname']



import faiss
import numpy as np

def create_lsh_index(embeddings, n_bits=256):
    """
    Create an LSH index for embeddings with faiss.

    Args:
    embeddings: 2D numpy array of shape (n_samples, n_features)
    n_bits: The number of bits to use for hashing in LSH. More bits, finer granularity.

    Returns:
    index: The faiss LSH index.
    """
    d = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexLSH(d, n_bits)
    index.add(embeddings)  # Add embeddings to the index
    return index

# Ensure embeddings are in the correct format (n_samples, n_features)
embeddings_array = np.ascontiguousarray(embeddings, dtype=np.float32)

# Create the LSH index
lsh_index = create_lsh_index(embeddings_array, n_bits=128)  # Adjust n_bits as needed



def query_lsh_index(query_embedding, index, n=5):
    """
    Query the LSH index for similar items.

    Args:
    query_embedding: Query embedding vector (1D numpy array).
    index: The faiss LSH index.
    n: Number of similar items to find.

    Returns:
    D: Distances of the n most similar items.
    I: Indices of the n most similar items in the dataset.
    """
    D, I = index.search(query_embedding.reshape(1, -1), n)  # Reshape query for single query
    return D, I

# Example query with the first embedding
D, I = query_lsh_index(embeddings_array[0], lsh_index, n=5)

# Example function to print similar scientists based on LSH query results
def print_similar_scientists(query_index, similar_indices, df):
    """
    Print the information of similar scientists based on the LSH query results.

    Args:
    query_index: The index of the query scientist in the DataFrame.
    similar_indices: Array of indices of similar scientists found by LSH.
    df: The pandas DataFrame containing the scientists' data.
    """
    print(f"Query Scientist: {df.iloc[query_index]['surname']} - {df.iloc[query_index]['education']}\n")
    print("Similar Scientists:")
    for idx in similar_indices[0]:  # Loop through the first row of indices returned by LSH
        # Skip the scientist itself if included in the results
        if idx != query_index:
            print(f"{df.iloc[idx]['surname']} with LSH distance: {lsh_distance:.2f} - {df.iloc[idx]['education']}")

query_index = matching_indices[0]
D, I = query_lsh_index(embeddings_array[0], lsh_index, n=5)  # Query for the first embedding


print_similar_scientists(query_index, I, df)


!pip install rtree
from rtree import index


class RTree:
    def __init__(self):
        self.idx = index.Index()
        self.data_list = []

    def insert(self, item_id, item, x, y):
        self.idx.insert(item_id, (x, y, x, y))
        self.data_list.append(item)

    def search(self, query_bbox):
        return list(self.idx.intersection(query_bbox))


def build_rtree():
    rtree = RTree()

    for i in range(len(df)):
        x = letter_normalization(df.iloc[i]['surname'][0])
        y = df.iloc[i]['awards']
        data = (df.iloc[i]['surname'], df.iloc[i]['awards'], df.iloc[i]['education'])
        rtree.insert(i, data, x, y)

    return rtree


def query_rtree(rtree, min_letter, max_letter, num_awards):
    min_letter = letter_normalization(min_letter)
    max_letter = letter_normalization(max_letter)

    query_bbox = (min_letter, num_awards, max_letter, float('inf'))
    matches = rtree.search(query_bbox)

    query_results = []
    for match in matches:
        surname, awards, education = rtree.data_list[match]
        query_results.append({"surname": surname, "awards": awards, "education": education})

    return query_results


# Step 1: Build R-tree and Query
rtree = build_rtree()
query_results = query_rtree(rtree, 'A', 'G', 4)  # Example query

# Step 2: Extract Text Data (Education Descriptions)
education_descriptions = [result['education'] for result in query_results]

# Step 3: Generate BERT Embeddings
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

embeddings = [embedding(text) for text in education_descriptions]


import faiss
import numpy as np


# Convert embeddings list to a NumPy array
embeddings_array = np.array(embeddings).astype('float32')

d = embeddings_array.shape[1]  # Dimensionality of embeddings
n_bits = 256  # Number of bits for LSH, adjust based on dataset size and desired precision

# Initialize LSH index
index = faiss.IndexLSH(d, n_bits)
index.add(embeddings_array)  # Add embeddings to the index

# Function to perform LSH query
def query_lsh(index, query_embedding, n=5):
    """
    Query the LSH index for similar embeddings.

    Args:
    - index: The LSH index.
    - query_embedding: The query embedding vector.
    - n: Number of nearest neighbors to retrieve.

    Returns:
    - D: Distances of the nearest neighbors.
    - I: Indices of the nearest neighbors.
    """
    D, I = index.search(query_embedding.reshape(1, -1), n)
    return D, I

# Example query with the first embedding
D, I = query_lsh(index, embeddings_array[0], n=5)


def print_similar_scientists(query_index, similar_indices, distances, query_results):
    """
    Print information for scientists similar to the query scientist, including similarity scores.

    Args:
    - query_index: Index of the query scientist in the query_results list.
    - similar_indices: Indices of similar scientists found by LSH.
    - distances: Array of distances corresponding to the similar scientists.
    - query_results: List of dictionaries containing scientists' information.
    """
    print(f"Query Scientist: {query_results[query_index]['surname']} - {query_results[query_index]['education']}")
    print("Similar Scientists:")
    for i, index in enumerate(similar_indices[0]):
        if index != query_index:  # Avoid printing the query scientist itself
            lsh_distance = distances[0][i]
            print(f"{query_results[index]['surname']} with LSH distance: {lsh_distance:.2f} - {query_results[index]['education']}")

# You would then call the function with the distances as follows:
print_similar_scientists(0, I, D, query_results)



class Node1D:
    def __init__(self, y, i_list):
        self.y = y
        self.i_list = i_list
        self.left = None
        self.right = None
        self.height = 1

    def merge_i_list(self, i_list):
        self.i_list.extend(i_list)
        self.i_list = list(set(self.i_list))


class RangeTree1D:
    def __init__(self, points):
        self.root = self.build(points)

    def insert(self, root, y, i):

        if not root:
            return Node1D(y, [i])
        if y == root.y:
            root.merge_i_list([i])
        elif y < root.y:
            root.left = self.insert(root.left, y, i)
        else:
            root.right = self.insert(root.right, y, i)


        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)


        if balance > 1:
            if y > root.left.y:
                root.left = self.left_rotate(root.left)
            return self.right_rotate(root)


        if balance < -1:
            if y < root.right.y:
                root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def build(self, points):
        root = None
        for _, y, i in points:
            root = self.insert(root, y, i)
        return root


    def get_height(self, node):
        if not node:
            return 0
        return node.height


    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)


    def right_rotate(self, y):
        x = y.left
        T3 = x.right
        x.right = y
        y.left = T3
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        return x


    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        return y


    def query(self, node, y1, y2, result):
        if not node:
            return
        if y1 <= node.y <= y2:
            for i in node.i_list:
                result.append((node.y, i))
        if y1 < node.y:
            self.query(node.left, y1, y2, result)
        if y2 > node.y:
            self.query(node.right, y1, y2, result)


class Node2D:
    def __init__(self, x, points):
        self.x = x
        self.y_tree = RangeTree1D(points)
        self.left = None
        self.right = None
        self.height = 1

    def merge_point(self, y, i):
        self.y_tree.insert(self.y_tree.root, y, i)


class RangeTree2D:
    def __init__(self, points):
        self.root = self.build(points)

    def insert(self, root, x, y, i, points):
        if not root:
            return Node2D(x, [(x, y, i)])
        if x == root.x:
            root.y_tree.root = root.y_tree.insert(root.y_tree.root, y, i)
        elif x < root.x:
            root.left = self.insert(root.left, x, y, i, [(x, y, i)])
        else:
            root.right = self.insert(root.right, x, y, i, [(x, y, i)])

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))


        balance = self.get_balance(root)

        if balance > 1:
            if x > root.left.x:
                root.left = self.left_rotate(root.left)
            return self.right_rotate(root)


        if balance < -1:
            if x < root.right.x:
                root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def build(self, points):
        root = None
        for point in points:
            x, y, i = point
            root = self.insert(root, x, y, i, [point])
        return root

    def get_height(self, node):
        if not node:
            return 0
        return node.height

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)


    def right_rotate(self, y):
        x = y.left
        T3 = x.right
        x.right = y
        y.left = T3
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = max(self.get_height(x.left), self.get_height(x.right)) + 1
        y.height = max(self.get_height(y.left), self.get_height(y.right)) + 1
        return y

    def query(self, node, x1, x2, y1, y2, result):
        if not node:
            return
        if x1 <= node.x <= x2:
            y_result = []
            node.y_tree.query(node.y_tree.root, y1, y2, y_result)
            for y, i in y_result:
                result.append((node.x, y, i))
        if x1 < node.x:
            self.query(node.left, x1, x2, y1, y2, result)
        if x2 > node.x:
            self.query(node.right, x1, x2, y1, y2, result)


def build_range_tree():
    points = []

    for i in range(len(df)):
        x = letter_normalization(df.iloc[i]['surname'][0])
        y = df.iloc[i]['awards']
        points.append((x, y, i))

    range_tree = RangeTree2D(points)
    return range_tree


def query_range_tree(range_tree, min_letter, max_letter, num_awards):
    min_letter = letter_normalization(min_letter)
    max_letter = letter_normalization(max_letter)

    x_range = (min_letter, max_letter)
    y_range = (num_awards, float('inf'))

    query_results = []
    range_tree.query(range_tree.root, x_range[0], x_range[1], y_range[0], y_range[1], query_results)

    final_results = []
    for result in query_results:
        index = result[2]
        surname = df.iloc[index]['surname']
        awards = df.iloc[index]['awards']
        education = df.iloc[index]['education']
        final_results.append({"surname": surname, "awards": awards, "education": education})

    return final_results


range_tree = build_range_tree()
min_letter = 'A'
max_letter = 'G'
num_awards = 4
query_results = query_range_tree(range_tree, min_letter, max_letter, num_awards)
education_descriptions = [result['education'] for result in query_results]


from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

embeddings = np.array([embedding(text) for text in education_descriptions])


# Convert embeddings to a compatible format for Faiss, initialize LSH index, and add embeddings
embeddings_array = embeddings.astype('float32')
index = faiss.IndexLSH(embeddings_array.shape[1], 128)
index.add(embeddings_array)

# Query LSH index for a given embedding
D, I = query_lsh(index, embeddings_array[0], n=5)


for i in range(len(I[0])):
    idx = I[0][i]
    if idx != 0:  # skip the same scientist
        similar_scientist = query_results[idx]
        # Retrieve the LSH distance for the current index
        lsh_distance = D[0][i]
        print(f"Similar Scientist to {query_results[0]['surname']} with LSH distance: {lsh_distance:.2f}: {similar_scientist['surname']} - {similar_scientist['education']} ")


