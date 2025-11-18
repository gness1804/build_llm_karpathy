# Reddit Data Collection Setup

## Quick Setup (5 minutes)

### Step 1: Install PRAW Library
```bash
pip install praw
```

### Step 2: Create Reddit App
1. Go to https://www.reddit.com/prefs/apps
2. Scroll down and click **"create another app..."** or **"create app"**
3. Fill in:
   - **Name**: "LLM Training Data Collector" (or any name)
   - **Type**: Select **"script"**
   - **Description**: "Collecting data for LLM training" (optional)
   - **Redirect URI**: `http://localhost:8080` (required but not used)
4. Click **"create app"**

### Step 3: Get Credentials
After creating the app, you'll see:
- **Client ID**: The string under your app name (looks like: `abc123def456ghi`)
- **Client Secret**: The "secret" field (looks like: `xyz789_secret_key_here`)

### Step 4: Set Environment Variables
```bash
export REDDIT_CLIENT_ID='your_client_id_here'
export REDDIT_CLIENT_SECRET='your_client_secret_here'
export REDDIT_USER_AGENT='LLM Training Data Collector 1.0'
```

**To make permanent**, add to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export REDDIT_CLIENT_ID="your_client_id_here"' >> ~/.zshrc
echo 'export REDDIT_CLIENT_SECRET="your_client_secret_here"' >> ~/.zshrc
echo 'export REDDIT_USER_AGENT="LLM Training Data Collector 1.0"' >> ~/.zshrc
source ~/.zshrc
```

### Step 5: Test Collection
```bash
cd ~/Desktop/build_llm_karpathy/sources
python3 collect_reddit_data.py --limit 10
```

You should see:
```
✅ Connected to Reddit API
Collecting from r/relationships...
  Limit: 10 posts
  ...
✅ Collected X posts
```

## Usage Examples

### Basic Collection (100 posts)
```bash
python3 collect_reddit_data.py --limit 100
```

### High-Quality Collection (500 top posts, min 50 upvotes)
```bash
python3 collect_reddit_data.py \
  --limit 500 \
  --min-upvotes 50 \
  --sort top \
  --time-filter all
```

### Quick Test (10 posts, lower quality threshold)
```bash
python3 collect_reddit_data.py \
  --limit 10 \
  --min-upvotes 5 \
  --min-comment-length 50
```

### Custom Output File
```bash
python3 collect_reddit_data.py \
  --limit 200 \
  --output reddit_data_large.md
```

## Recommended Settings

### For Initial Testing
```bash
python3 collect_reddit_data.py \
  --limit 50 \
  --min-upvotes 10 \
  --min-comment-length 100
```
**Result**: ~1-2MB, takes 2-3 minutes

### For Training Dataset (5MB target)
```bash
python3 collect_reddit_data.py \
  --limit 500 \
  --min-upvotes 20 \
  --min-comment-length 150 \
  --sort top \
  --time-filter all
```
**Result**: ~5-10MB, takes 10-15 minutes

### For Large Dataset (10MB+)
```bash
python3 collect_reddit_data.py \
  --limit 1000 \
  --min-upvotes 15 \
  --min-comment-length 100 \
  --sort top \
  --time-filter all
```
**Result**: ~10-20MB, takes 20-30 minutes

## Combining with Carolyn Hax Data

After collecting Reddit data:

```bash
# 1. Clean Reddit data (optional, removes metadata)
python3 clean_carolyn_hax_data.py reddit_relationships_*.md -o reddit_relationships_cleaned.md

# 2. Merge with Carolyn Hax data
# (Manually combine files or create a new merge script)

# 3. Or use Reddit data separately
# Just point training to the Reddit file
```

## Troubleshooting

### "Reddit API credentials not found"
- Make sure you set the environment variables
- Run `echo $REDDIT_CLIENT_ID` to verify
- Restart terminal after adding to `.zshrc`

### "Rate limit exceeded"
- Reddit API allows 60 requests/minute
- The script respects rate limits automatically
- If you hit limits, wait a few minutes and try again

### "No posts collected"
- Lower `--min-upvotes` (try 5 instead of 10)
- Lower `--min-comment-length` (try 50 instead of 100)
- Try different `--sort` method (hot, new, top)
- Check if subreddit exists: `--subreddit relationships`

### Connection Errors
- Check internet connection
- Verify Reddit is accessible
- Try again in a few minutes (Reddit may be down)

## Alternative: Other Subreddits

You can collect from other advice subreddits:

```bash
# Relationship advice
python3 collect_reddit_data.py --subreddit relationships

# General advice
python3 collect_reddit_data.py --subreddit Advice

# Personal finance advice
python3 collect_reddit_data.py --subreddit personalfinance

# Career advice
python3 collect_reddit_data.py --subreddit careers
```

## Next Steps

1. **Collect initial dataset**: Start with 100-200 posts
2. **Review quality**: Check the output file
3. **Adjust filters**: Tune min-upvotes and comment-length
4. **Collect more**: Scale up to 500-1000 posts for training
5. **Clean and merge**: Combine with existing data or use separately

