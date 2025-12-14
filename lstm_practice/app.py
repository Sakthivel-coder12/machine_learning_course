import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('shakespeare_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word with better handling
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        # Tokenize the input text
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        # Debug info
        st.write(f"Tokenized text: {token_list}")
        
        # Handle empty token list
        if not token_list:
            st.warning("No tokens found. Try different words.")
            return None
            
        # Trim if too long
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        st.write(f"Padded sequence shape: {token_list.shape}")
        
        # Get predictions
        predicted = model.predict(token_list, verbose=0)
        st.write(f"Predictions shape: {predicted.shape}")
        
        # Get the index of the highest probability word
        # FIX: np.argmax returns a scalar when axis=1, need to get the first element
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        # Get the probability for debugging
        predicted_prob = np.max(predicted)
        st.write(f"Predicted index: {predicted_word_index}, Probability: {predicted_prob:.4f}")
        
        # Check if index is valid
        if predicted_word_index >= len(tokenizer.word_index):
            st.warning(f"Predicted index {predicted_word_index} is out of vocabulary range (0-{len(tokenizer.word_index)-1})")
            return "<OOV>"
        
        # Find the word for this index
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        
        return "<OOV>"
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Alternative function with temperature sampling
def predict_next_word_with_temperature(model, tokenizer, text, max_sequence_len, temperature=1.0):
    try:
        # Tokenize
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        if not token_list:
            return None
            
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Get predictions
        predictions = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature for more interesting results
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # Get top 5 predictions for debugging
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        st.write("**Top 5 predictions:**")
        for idx in top_indices:
            word = None
            for w, i in tokenizer.word_index.items():
                if i == idx:
                    word = w
                    break
            if word:
                prob = predictions[idx]
                st.write(f"  - {word}: {prob:.3%}")
        
        # Sample from the distribution
        predicted_word_index = np.random.choice(len(predictions), p=predictions)
        
        # Find the word
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        
        return "<OOV>"
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit app
st.title("Next Word Prediction With LSTM")
st.markdown("---")

# Get model info
try:
    max_sequence_len = model.input_shape[1] + 1
    vocab_size = len(tokenizer.word_index)
    
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Input sequence length: {max_sequence_len - 1}")
    st.sidebar.write(f"Vocabulary size: {vocab_size}")
    st.sidebar.write(f"Model input shape: {model.input_shape}")
    st.sidebar.write(f"Model output shape: {model.output_shape}")
    
    # Show sample vocabulary
    if st.sidebar.checkbox("Show vocabulary sample"):
        sample_words = list(tokenizer.word_index.keys())[:50]
        st.sidebar.write(f"Sample words: {', '.join(sample_words)}")
        
except Exception as e:
    st.sidebar.error(f"Error getting model info: {e}")

# Input section
input_text = st.text_input("Enter text:", "To be or not to")

col1, col2, col3 = st.columns(3)

with col1:
    predict_simple = st.button("Predict Next Word", type="primary")

with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, key="temp_slider")

with col3:
    predict_temp = st.button("Predict with Temperature")

# Prediction results
if predict_simple or predict_temp:
    st.markdown("---")
    
    if not input_text.strip():
        st.warning("Please enter some text!")
    else:
        try:
            if predict_simple:
                st.subheader("Simple Prediction")
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            else:
                st.subheader("Temperature-based Prediction")
                next_word = predict_next_word_with_temperature(model, tokenizer, input_text, max_sequence_len, temperature)
            
            if next_word:
                if next_word == "<OOV>":
                    st.warning("⚠️ Predicted word is Out Of Vocabulary")
                    st.write("Try a different input or use temperature sampling.")
                else:
                    st.success(f"**Next word:** {next_word}")
                    
                    # Show the full sentence
                    st.info(f"**Full sentence:** {input_text} **{next_word}**")
                    
                    # Show similar words
                    if st.checkbox("Show word context"):
                        try:
                            # Find words that often appear with the predicted word
                            st.write("Words that often appear together:")
                            # You could add more sophisticated context analysis here
                        except:
                            pass
            else:
                st.error("Could not make prediction.")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Try these examples
st.markdown("---")
st.subheader("Try these examples:")

examples = [
    "to be or not to",
    "shall i compare thee",
    "romeo wherefore art",
    "all the world is",
    "if music be the"
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(example, key=f"ex_{i}"):
        st.session_state.input_text = example
        st.rerun()

# Debug section (collapsible)
with st.expander("Debug Information"):
    st.write("### Tokenizer Information")
    st.write(f"Total words in tokenizer: {len(tokenizer.word_index)}")
    
    # Check if specific words are in vocabulary
    test_words = ["to", "be", "or", "not", "the", "and", "i", "you", "my", "a"]
    missing_words = []
    
    st.write("### Vocabulary Check")
    for word in test_words:
        if word in tokenizer.word_index:
            st.write(f"✓ '{word}' is in vocabulary (index: {tokenizer.word_index[word]})")
        else:
            st.write(f"✗ '{word}' is NOT in vocabulary")
            missing_words.append(word)
    
    if missing_words:
        st.warning(f"Missing common words: {missing_words}")
    
    # Test prediction with known words
    st.write("### Test Prediction")
    test_text = "to be or"
    test_tokens = tokenizer.texts_to_sequences([test_text])[0]
    st.write(f"Test text: '{test_text}'")
    st.write(f"Tokenized: {test_tokens}")
    
    if test_tokens:
        test_padded = pad_sequences([test_tokens], maxlen=max_sequence_len-1, padding='pre')
        st.write(f"Padded shape: {test_padded.shape}")