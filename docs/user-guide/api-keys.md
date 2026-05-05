# API Keys

Scene Ripper uses cloud APIs for AI-powered features like clip descriptions, narrative generation, and text extraction. This guide walks you through getting an API key from each supported provider and entering it in the app.

## How API Keys Work in Scene Ripper

API keys are stored securely in your system's keyring (macOS Keychain, Windows Credential Manager, or Linux Secret Service). You enter them once in **Settings > API Keys** and they persist across sessions.

Environment variables take priority over stored keys. If you set `ANTHROPIC_API_KEY` in your shell, Scene Ripper will use that instead of whatever is saved in settings.

## Which Keys Do I Need?

You don't need all of these. Most users need just one or two.

| Feature | What You Need |
|---------|--------------|
| Chat agent | Any LLM provider: Anthropic, OpenAI, Gemini, or OpenRouter |
| Describe clips | Any VLM provider: Gemini, OpenAI, or Anthropic |
| Classify shots | Any VLM provider |
| Extract on-screen text | Any VLM provider |
| Storyteller sequencer | Any LLM provider |
| Exquisite Corpus sequencer | Any LLM provider for poem generation; OCR can run locally or use a VLM fallback |
| Free Association sequencer | Any LLM provider |
| Signature Style sequencer (VLM mode) | Any VLM provider |
| YouTube search and download | YouTube Data API |
| Replicate models | Replicate |

**Recommendation:** A Gemini key is the most versatile starting point. It works as both an LLM and VLM provider, and Google offers a generous free tier.

---

## Anthropic

Anthropic makes the Claude family of models. Claude is particularly strong at narrative and creative writing tasks like the Storyteller sequencer.

### Create an Account

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Click **Sign Up** and create an account with your email or Google account
3. Verify your email address

### Generate an API Key

1. Once logged in, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
2. Click **Create Key**
3. Give the key a name (e.g., "Scene Ripper")
4. Copy the key immediately. It starts with `sk-ant-` and won't be shown again

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Paste your key in the **Anthropic** field
4. Click **Save**

### Environment Variable Alternative

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Pricing

Anthropic offers a pay-as-you-go model. New accounts receive a small amount of free credits. See [anthropic.com/pricing](https://www.anthropic.com/pricing) for current rates.

---

## OpenAI

OpenAI makes the GPT family of models. GPT models work well for both text and vision tasks in Scene Ripper.

### Create an Account

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Click **Sign Up** and create an account
3. Verify your email and phone number

### Generate an API Key

1. Once logged in, go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **Create new secret key**
3. Give it a name (e.g., "Scene Ripper")
4. Copy the key immediately. It starts with `sk-` and won't be shown again

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Paste your key in the **OpenAI** field
4. Click **Save**

### Environment Variable Alternative

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Pricing

OpenAI uses pay-as-you-go pricing. New accounts may receive free credits. See [openai.com/pricing](https://openai.com/api/pricing/) for current rates.

---

## Google Gemini

Google's Gemini models are strong at vision tasks (describing clips, classifying shots, extracting text) and offer a generous free tier.

### Create an Account

1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Sign in with your Google account
3. Accept the terms of service

### Generate an API Key

1. Once signed in, go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Select a Google Cloud project (or create a new one if prompted)
4. Copy the key. It starts with `AIza`

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Paste your key in the **Gemini** field
4. Click **Save**

### Environment Variable Alternative

```bash
export GEMINI_API_KEY=AIza-your-key-here
```

### Pricing

Gemini offers a free tier with generous rate limits for most models. Paid usage is billed through Google Cloud. See [ai.google.dev/pricing](https://ai.google.dev/pricing) for current details.

---

## OpenRouter

OpenRouter is a unified API that gives you access to models from multiple providers (Anthropic, OpenAI, Google, Meta, and others) through a single key. Useful if you want to try different models without managing separate accounts.

### Create an Account

1. Go to [openrouter.ai](https://openrouter.ai/)
2. Click **Sign Up** and create an account
3. Add credits to your account (OpenRouter is prepaid)

### Generate an API Key

1. Once logged in, go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Click **Create Key**
3. Give it a name (e.g., "Scene Ripper")
4. Copy the key. It starts with `sk-or-`

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Paste your key in the **OpenRouter** field
4. Click **Save**

### Environment Variable Alternative

```bash
export OPENROUTER_API_KEY=sk-or-your-key-here
```

### Pricing

OpenRouter charges per-token based on the underlying model's pricing, plus a small markup. See [openrouter.ai/models](https://openrouter.ai/models) for per-model pricing.

---

## Replicate

Replicate hosts open-source models as cloud APIs. Used for specific model-based features in Scene Ripper.

### Create an Account

1. Go to [replicate.com](https://replicate.com/)
2. Click **Sign In** and create an account with GitHub or Google
3. Add a payment method (required for API access beyond the free tier)

### Generate an API Key

1. Once logged in, go to [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
2. Click **Create Token**
3. Copy the token. It starts with `r8_`

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Paste your token in the **Replicate** field
4. Click **Save**

### Environment Variable Alternative

> **Note:** The Replicate environment variable is `REPLICATE_API_TOKEN` (not `REPLICATE_API_KEY`). This follows Replicate's own naming convention.

```bash
export REPLICATE_API_TOKEN=r8_your-token-here
```

### Pricing

Replicate offers a small free tier for new accounts. After that, pricing varies by model and is billed per second of compute time. See [replicate.com/pricing](https://replicate.com/pricing) for details.

---

## YouTube Data API

The YouTube API key enables searching for videos and fetching metadata directly in Scene Ripper's Collect tab. This setup is more involved than the other providers because it goes through Google Cloud Console.

### Create a Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Click the project dropdown at the top of the page and select **New Project**
4. Name it something like "Scene Ripper" and click **Create**
5. Wait a moment for the project to be created, then select it from the project dropdown

### Enable the YouTube Data API

1. In the left sidebar, go to **APIs & Services > Library**
2. Search for **YouTube Data API v3**
3. Click on it and then click **Enable**

### Generate an API Key

1. In the left sidebar, go to **APIs & Services > Credentials**
2. Click **Create Credentials** at the top and select **API key**
3. A dialog will show your new key. Copy it immediately
4. (Optional but recommended) Click **Restrict Key** to limit it to only the YouTube Data API v3

### Enter in Scene Ripper

1. Open **Settings** (Cmd+, on macOS, or File > Settings)
2. Go to the **API Keys** tab
3. Scroll down to the **YouTube Data API** section
4. Paste your key in the API key field
5. Click **Save**

### Environment Variable Alternative

```bash
export YOUTUBE_API_KEY=your-key-here
```

### Pricing

The YouTube Data API v3 includes a daily quota of 10,000 units per project at no cost. Each search request costs about 100 units, so you get roughly 100 searches per day for free. This is sufficient for most Scene Ripper workflows. See [Google's quota documentation](https://developers.google.com/youtube/v3/getting-started#quota) for details.
