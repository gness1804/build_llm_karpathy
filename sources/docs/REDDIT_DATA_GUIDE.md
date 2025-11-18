# Using Reddit Data for Training

## Why Reddit r/relationships is a Great Option

### Advantages
1. **Massive dataset**: r/relationships has millions of posts spanning years
2. **Similar format**: Q&A style (problem → advice) like Carolyn Hax
3. **Public data**: Reddit posts are public and generally acceptable for training
4. **Easy to collect**: Reddit API and existing tools make collection straightforward
5. **Diverse content**: Wide variety of relationship scenarios
6. **Free**: No copyright concerns (user-generated content)

### Comparison to Carolyn Hax

| Aspect | Carolyn Hax | Reddit r/relationships |
|--------|-------------|------------------------|
| **Size** | ~750KB (limited) | Millions of posts (unlimited) |
| **Quality** | Professional, curated | Variable (some excellent, some not) |
| **Format** | Consistent Q&A | Similar Q&A format |
| **Collection** | Manual, time-intensive | Automated, fast |
| **Copyright** | Copyrighted (Washington Post) | Public, user-generated |
| **Style** | Professional advice columnist | Community advice |

## Ethical & Legal Considerations

### ✅ Generally Acceptable
- **Public Reddit posts** are user-generated content posted publicly
- Reddit's Terms of Service allow accessing public content via API
- Using public posts for research/training is common practice
- Many academic datasets use Reddit data

### ⚠️ Best Practices
1. **Respect rate limits**: Use Reddit API properly (don't hammer servers)
2. **Filter content**: Remove personal information, NSFW content if desired
3. **Attribution**: Consider noting data source in your model documentation
4. **Privacy**: Don't include usernames or identifying info in training data
5. **Content filtering**: Filter out low-quality or off-topic posts

### 📝 Reddit API Terms
- Reddit API is free for reasonable use
- Rate limits: 60 requests per minute (with OAuth)
- Commercial use: Generally OK for public data
- Check current Reddit API terms for latest policies

## Data Quality Considerations

### Pros
- **Volume**: Can easily get 10-50MB+ of quality content
- **Diversity**: Many different relationship scenarios
- **Real-world**: Authentic problems and advice
- **Community wisdom**: Often includes multiple perspectives

### Cons
- **Variable quality**: Some advice is excellent, some is poor
- **Noise**: May include off-topic or low-effort posts
- **Formatting**: Need to clean and standardize format
- **Tone**: More casual than professional advice columnist

### Quality Filtering Strategies
1. **Upvote threshold**: Only use posts with 10+ upvotes (indicates quality)
2. **Comment length**: Filter for substantial advice (100+ words)
3. **Subreddit rules**: r/relationships has moderation, so quality is decent
4. **Date range**: Recent posts may be more relevant
5. **Remove deleted/removed**: Filter out deleted posts

## Collection Methods

### Option 1: Reddit API (Recommended)
- Official, well-documented
- Rate-limited but sufficient
- Requires API credentials (free)
- Best for ongoing collection

### Option 2: Pushshift Archive
- Historical Reddit data archive
- Pre-collected datasets available
- May have older data
- Check availability/access

### Option 3: Web Scraping
- Can work but less reliable
- May violate ToS if not careful
- Not recommended vs. API

## Format Conversion

Reddit format:
```
Title: [Problem description]
Post: [Full question/details]
Comments: [Advice responses]
```

Convert to:
```
QUESTION: [Combined title + post]
ANSWER: [Top/best advice comment]
```

## Recommended Approach

1. **Start with Reddit API**: Collect 1000-5000 top posts from r/relationships
2. **Filter for quality**: Upvotes > 10, substantial comments
3. **Format conversion**: Convert to QUESTION/ANSWER format
4. **Combine with Carolyn Hax**: Mix both datasets for training
5. **Evaluate**: Test if Reddit data improves model quality

## Quick Start

See `collect_reddit_data.py` for automated collection script.

