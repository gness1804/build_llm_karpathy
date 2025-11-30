# Alternative Data Sources (No Reddit API Needed)

Since Reddit API app creation is restricted, here are alternatives:

## Option 1: Reddit Public JSON (No Auth) ✅ RECOMMENDED

**Script**: `collect_reddit_data_no_auth.py`

**Pros**:
- ✅ No API credentials needed
- ✅ Works immediately
- ✅ Free and legal
- ✅ Same data quality

**Cons**:
- ⚠️ Slower (rate limiting)
- ⚠️ Limited to ~25 posts per request

**Usage**:
```bash
python3 sources/scripts/collect_reddit_data_no_auth.py --limit 50
```

**Time**: ~5-10 minutes for 50 posts (~1-2MB)

## Option 2: HuggingFace Datasets

Pre-collected Reddit datasets available on HuggingFace:

### Reddit Relationships Dataset
```python
from datasets import load_dataset

dataset = load_dataset("reddit", "relationships")
# Or search: https://huggingface.co/datasets?search=reddit
```

**Pros**:
- ✅ Already collected and formatted
- ✅ Large datasets available
- ✅ No API needed

**Cons**:
- ⚠️ May not be up-to-date
- ⚠️ Need to convert format

## Option 3: Pushshift Archive

Historical Reddit data archive (may be restricted now):

- **Website**: https://pushshift.io/
- **Status**: May require access/API key
- **Data**: Historical Reddit posts

## Option 4: Web Scraping (Advanced)

Use BeautifulSoup or similar to scrape Reddit HTML:

**Pros**:
- ✅ No API needed
- ✅ Full control

**Cons**:
- ⚠️ May violate ToS
- ⚠️ Fragile (breaks if HTML changes)
- ⚠️ More complex

**Not recommended** - use public JSON instead.

## Option 5: Existing Datasets

### Academic Datasets
- **Reddit Dataset**: Various academic collections
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- Search: "reddit relationships dataset"

### Kaggle Datasets
- Search Kaggle for "reddit relationships"
- Some require account, but free

## Option 6: RSS Feeds

Reddit provides RSS feeds (no auth needed):

```
https://www.reddit.com/r/relationships/top/.rss?limit=25
```

**Pros**:
- ✅ Simple
- ✅ No auth

**Cons**:
- ⚠️ Limited data (no comments easily)
- ⚠️ Need to parse RSS

## Recommendation

**Use Option 1** (`collect_reddit_data_no_auth.py`):
- Works immediately
- No setup needed
- Good enough for your use case
- Can run multiple times to collect more

**Quick Start**:
```bash
cd ~/Desktop/build_llm_karpathy
python3 sources/scripts/collect_reddit_data_no_auth.py --limit 100
```

This will take ~10-15 minutes and give you ~2-4MB of data.

## If You Still Want Reddit API Access

1. **Verify your Reddit account**: Make sure email is verified
2. **Contact Reddit Support**: Use the link in the error message
3. **Explain use case**: "Collecting public data for personal LLM training"
4. **Wait for approval**: Usually takes a few days

But honestly, the no-auth method works fine for your needs!

