import pandas as pd
import numpy as np
import random

# Step 1: Create the dataset
data = {
    "song_title": [
        "Aaruyire", 
        "Ennavale Adi Ennavale", 
        "Vennilave Vennilave", 
        "Unnai Kandu Naan", 
        "Yaaro Ivan Yaaro",
        "Mannavan Vanthanadi", 
        "Azhagiya Tamil Magan", 
        "Pudhu Vellai Mazhai", 
        "Thamarai Poovukkum", 
        "Suttrum Vizhi", 
        "Anbe Anbe", 
        "Oru Deivam Thandha Poove", 
        "Kadhal Rojave", 
        "Nenjukkul Peidhidum", 
        "Ennai Kolathey", 
        "Naan Yaar", 
        "Venpani Malargal", 
        "Oru Koodai Sunlight", 
        "Suttrum Boomi", 
        "Chinna Chinna Aasai", 
        "Vaadi Pulla Vaadi",         
        "Pakkam Vanthu",     
        "Yaaradi Nee Mohini",        
        "Kanna Veesi",   
        "Thuli Thuli",                 
        "Marana Mass",                
        "Ninaithale Inikkum",         
        "Kannana Kanne",            
        "Sakkarai Nilave",            
        "Pudhu Pudhu Arthangal"       
    ],
    "mood": [
        "happy", 
        "romantic", 
        "melancholic", 
        "sad", 
        "happy",
        "happy", 
        "romantic", 
        "happy", 
        "sad", 
        "melancholic", 
        "romantic", 
        "sad", 
        "romantic", 
        "happy", 
        "sad", 
        "sad", 
        "melancholic", 
        "happy", 
        "happy", 
        "motivational",
        "sad",                  
        "happy",                 
        "romantic",              
        "romantic",               
        "happy",                   
        "happy",                 
        "melancholic",         
        "romantic",             
        "happy",                  
        "motivational"           
    ]
}


# Create the DataFrame
df = pd.DataFrame(data)

# Step 2: Initialize the Q-learning variables
n_songs = len(df)
n_moods = len(df['mood'].unique())

# Initialize Q-table with zeros
Q = np.zeros((n_songs, n_moods))

# Define parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 1.0
exploration_decay = 0.99
min_exploration_prob = 0.1
n_episodes = 1000

# To track songs already recommended for each mood
recommended_songs = {mood: set() for mood in df['mood'].unique()}

# Step 3: Simulate User Interactions
def recommend_song(mood_index):
    # Get available songs for the mood that haven't been recommended yet
    available_songs = [i for i in range(n_songs) if df['mood'][i] == df['mood'].unique()[mood_index] and i not in recommended_songs[df['mood'].unique()[mood_index]]]
    
    # If no songs are available, reset the recommended songs for this mood
    if not available_songs:
        recommended_songs[df['mood'].unique()[mood_index]] = set()  # Reset
        available_songs = [i for i in range(n_songs) if df['mood'][i] == df['mood'].unique()[mood_index]]
    
    # Get the song index with the highest Q-value for the available songs
    song_index = max(available_songs, key=lambda x: Q[x, mood_index])
    return df.iloc[song_index]['song_title'], song_index

# Function to update the Q-table
def update_q_table(song_index, mood_index, reward):
    best_future_q = np.max(Q[song_index])
    Q[song_index, mood_index] += learning_rate * (reward + discount_factor * best_future_q - Q[song_index, mood_index])

# Step 4: Training the agent
for episode in range(n_episodes):
    # Randomly choose a mood index to simulate exploration
    mood_index = random.randint(0, n_moods - 1)
    
    # Recommend a song based on the chosen mood
    song_title, song_index = recommend_song(mood_index)

    # Simulate user feedback (randomly for this example)
    print(f"Recommended Song for mood '{df['mood'].unique()[mood_index]}': {song_title}")
    user_feedback = int(input("Rate the song (1-5): "))  # 1 is bad, 5 is excellent

    # Update the Q-table based on user feedback
    reward = user_feedback - 3  # Normalize feedback to -2 (bad) to +2 (excellent)
    update_q_table(song_index, mood_index, reward)

    # Track the recommended song for the mood
    recommended_songs[df['mood'].unique()[mood_index]].add(song_index)

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

# Step 5: Recommendations after training
print("\nRecommendations based on learned preferences:")
for mood_index in range(n_moods):
    song_title, _ = recommend_song(mood_index)
print(f"For mood '{df['mood'].unique()[mood_index]}', recommended song: {song_title}")
