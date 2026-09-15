import pandas as pd
import os

def load_and_filter_brand(filepath: str, brand_name: str = "AppleSupport") -> pd.DataFrame:
    """
    Loads the TWCS dataset and filters it to conversations involving a specific brand.
    Since we only need a sample, we will load in chunks to save memory if necessary.
    """
    print(f"Loading data from {filepath} for {brand_name}...")
    df = pd.read_csv(filepath)
    
    # Keep tweets where the brand is the author, OR the tweet is in response to the brand, OR the brand is tagged
    brand_tweets = df[df['author_id'] == brand_name]
    
    # Get the tweet IDs from the brand to find the inbound customer tweets
    brand_in_responses = brand_tweets['in_response_to_tweet_id'].dropna().unique()
    customer_tweets = df[df['tweet_id'].isin(brand_in_responses)]
    
    # Reconstruct simple 2-turn conversations (Customer -> Brand)
    merged = pd.merge(
        customer_tweets, 
        brand_tweets, 
        left_on='tweet_id', 
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_brand')
    )
    
    merged = merged[[
        'tweet_id_customer', 'text_customer', 'created_at_customer',
        'tweet_id_brand', 'text_brand', 'created_at_brand'
    ]]
    
    print(f"Reconstructed {len(merged)} conversations for {brand_name}")
    return merged

if __name__ == "__main__":
    if os.path.exists("twcs/twcs.csv"):
        df = load_and_filter_brand("twcs/twcs.csv")
        df.to_csv("apple_conversations.csv", index=False)
        print("Saved apple_conversations.csv")
    else:
        print("twcs/twcs.csv not found. Wait for download to finish.")
