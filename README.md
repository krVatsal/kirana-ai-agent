# Kirana AI Agent

A conversational AI assistant for Indian Kirana (grocery) stores. Customers and shopkeepers can interact via chat or voice, place orders, check inventory, and get real-time responses in Hindi, English, or Hinglish.]
Demo Video: https://drive.google.com/file/d/10gY_ajJ4q9VKKET4VSWUpbOeNluq5h42/view?usp=sharing

## Features

- 💬 Chatbox with text and speech-to-text input
- 🤖 AI-powered order processing and inventory queries
- 🌐 Multilingual support (Hindi, English, Hinglish)
- 📊 Shopkeeper dashboard for order and inventory management
- 🔊 Text-to-speech responses
- 💾 Persistent data storage with SQLite

## Technologies

- **Streamlit** - Web UI framework
- **Google Gemini** - AI reasoning and natural language processing
- **gTTS** - Text-to-speech conversion
- **SQLite** - Local data storage
- **Web Speech API** - Speech-to-text functionality

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krVatsal/kirana-ai-agent.git
   cd kirana-ai-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google Gemini API key in `.env`:**
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Customer Interface
- Type or speak messages to place orders or ask questions
- Support for natural language in Hindi, English, or mixed (Hinglish)
- Voice input using the microphone button 🎤
- Audio responses for accessibility

### Shopkeeper Dashboard
- View and manage orders in real-time
- Monitor inventory levels with low-stock alerts
- Track order statuses (processing → out-for-delivery → delivered)
- View sales metrics and revenue

## Project Structure

```
├── app.py          # Main Streamlit application
├── storage.py      # Database operations and data persistence
├── data.db         # SQLite database (created automatically)
├── pyproject.toml  # Project configuration
└── README.md       # Project documentation
```

## Key Features

### AI Capabilities
- Natural language understanding for order processing
- Intent recognition (order, inventory check, status inquiry, greeting)
- Smart inventory management with availability checking
- Multilingual response generation

### Voice Features
- Speech-to-text using Web Speech API
- Text-to-speech responses with language detection
- Real-time voice input processing

### Data Management
- Order tracking with unique IDs
- Inventory management with stock levels
- Chat history persistence
- Order status updates

## API Requirements

You'll need a Google Gemini API key to use the AI features. Get one from:
- [Google AI Studio](https://makersuite.google.com/app/apikey)

## Browser Compatibility

Speech-to-text features work best with:
- Google Chrome
- Microsoft Edge
- Other Chromium-based browsers

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

