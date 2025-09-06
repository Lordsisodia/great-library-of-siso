# I Tested Every New AI Model for Client Work - Here's the Winner

## Video Information
- **Source:** https://www.youtube.com/watch?v=RLoAirQpKL4
- **Date:** Not specified (referenced as recent GPT5 release)
- **Channel:** Not specified
- **Duration:** Approximately 12 minutes

## Key Topics
- Real-world AI agent performance comparison
- GPT5 vs Gemini vs Claude vs Grok testing
- Database query agent implementation
- Deep research agent capabilities
- Newsletter generation from Figma to HTML
- Cost-effectiveness analysis
- Token usage comparison
- Latency and speed benchmarks

## Key Takeaways

### Model Performance Summary
1. **GPT5**: Best for analytics and data analysis tasks, excellent at following instructions precisely, significantly reduced hallucinations, but suffers from high latency and token usage
2. **Claude Sonnet**: Superior for coding tasks, sometimes provides additional helpful solutions beyond instructions
3. **Gemini 2.5 Pro**: Best overall for general agentic use cases, balancing performance, speed, and cost
4. **Gemini 2.5 Flash**: Optimal for web browsing and research tasks, extremely cost-effective with only 30,000 tokens vs GPT5's 1M+ tokens
5. **Claude Opus**: Failed badly on research tasks, expensive at $7.50 for single newsletter generation
6. **Grok**: Consistently underperformed across all metrics

### Use Case Results
- **DB Query Agent**: GPT5 performed best but with terrible latency; Gemini 2.5 Pro won overall (40% performance, 30% speed, 30% cost weighting)
- **Deep Research Agent**: GPT5 scored 9/10 but used over 1 million tokens; Gemini 2.5 Flash won overall with 30,000 tokens
- **Newsletter Generation**: GPT5 produced near-perfect UI matching, Claude Sonnet also performed well, Grok failed completely

### Cost Analysis
- Claude Opus: Most expensive at $7.50 for newsletter generation
- GPT5: Cheaper per token but high usage negates savings
- Gemini models: Most cost-effective overall
- Manual work comparison: AI costs ($7.50) vs human labor ($100-$200)

### Recommendations by Use Case
- **Analytics/RAG**: GPT5
- **Coding**: Claude Sonnet
- **General Agents**: Gemini 2.5 Pro
- **Web Browsing/Research**: Gemini 2.5 Flash

## Full Transcript

(00:00) So, does GPT5 actually suck? Well, of course, it looks like a really good model on benchmarks, but as we all know, benchmarks don't really reflect the performance on real world tasks. And so, in this video, I built three real world AI agent use cases that we actually encounter quite frequently with our own clients.
(00:21) All of these use cases require complex multi-step reasoning and workflows with at least 10 to 25 tool calls for every single query. And so today we're going to compare GPT5 side by side on these real AI agent use cases with all other models like Gemini, Antropic and Grock. And to be honest, the results might surprise you. So here are the use cases that we are going to cover today.
(00:45) The first one is a DB query agent. So this is a very common analytics use case. The second one is a deep research agent which I already showed in one of my previous videos. This agent is made to perform research on both local files and on the web. And the last agent is a newsletter generation agent which converts Figma files into HTML.
(01:02) So all of these use cases are honestly extremely valuable. And all of them are actually a lot more complicated than it might seem at first. Like for example, the DBQY agent is actually made to work with databases that have hundreds of tables and not just hundreds of tables, but hundreds of unlabelled tables without clear descriptions, which is something that we actually encounter quite frequently with enterprises.
(01:23) So again, this agent has a really complicated workflow. However, it does have a very simple structure where it is just one DB query agent with three tools. So, all of these agents are built using the new version of our framework which allows you to easily switch between different LLMs with light LLM. So, here we have our light LM config.
(01:40) And as I said, we're going to be testing all of the most common models that you can think of. And for the SQL queries themselves, we're going to be testing questions like what is the company's gross revenue from specific date to specific date. And although this question might seem simple, again to answer this question with a SQL database like this requires a lot of work.
(01:57) And this is something that our team has been working on for a very long time for one specific company in order to get right. Because in order for an agent to estimate the company's gross revenue, it needs to fetch this data from multiple different places. And to evaluate the performance of these agents, we're going to use another scoring agent.
(02:14) And this agent essentially is going to evaluate whether the responses produced by the models align with the criteria. and also whether the results of those queries actually match the expected numbers. So now let's run the model comparison and let's see the results. All right, so now around an hour later the results are finally completed.
(02:34) It took honestly a very long time because again we were running seven models on three different queries. And as you can see the first thing that stands out is that actually anthropic clot oppus spent almost on this task which is significantly higher than any other model. However, the LLM score is actually not that big.
(02:52) So, let me open the comparison charts right here and let's actually see this visually. So, here we have four different charts that compare the performance, the speed, and the cost effectiveness of each model. What's important to note is that the higher the better even for cost effectiveness.
(03:10) And as you can see, GPT5 actually performed the best on this use case in terms of performance. I actually have noticed before that GPT5 is exceptional at analyzing the data. However, the latency of this model is absolutely horrible. So, this is also one thing that I noticed from my testing before is that it takes way too long to think sometimes even for simple tasks.
(03:29) What's also interesting is that GPT5 mini is almost the same in terms of performance. However, the costs actually do not differ too much. And the reason why the costs don't differ that much is because if we open this next chart, we can see that GPT5 Mini actually spent double the tokens of GBT5 in order to complete that task. while Gemini 2.
(03:49) 5 flash spent almost eight times less than GBT 5 mini and Grock unfortunately didn't stand out uh in any of the metrics. So now let's look at the overall comparison. The overall grade is actually the best for Gemini 2.5 Pro. So the way we rank the overall grade is we give 40% for the performance, 30% for speed and 30% for cost.
(04:13) So of course for your specific use case this might be different. For some use cases, you might want to prioritize performance more than you want to prioritize speed. However, right now, it seems like Gemini 2.5 Pro is actually the best for analytics use cases because it provides a decent performance with very high speed and very high cost effectiveness.
(04:32) Now, let's move to the second use case, which is the deep research agent. This agent uses essentially one MCP server and it also uses a web search tool. What's very cool about light LLM is that it actually uses a native web search tool for a specific model provider. So for example, if we use Gemini here, the web search tool is not going to be by OpenI, it's actually going to be by Google.
(04:53) And the same applies to Grock and Antropic. So this way we actually have a more fair comparison of all these model providers. It's not just comparing the model, it's actually comparing the built-in tools for these providers as well. And the research query for this task is to compare langraph versus openi agents SDK versus crewi versus pientic AI for production AI agents.
(05:14) And for the files we're uploading all of those framework documentations. So as you can see the files for this task are absolutely massive. This is the complete documentation for each framework. So now let's run the model comparison and let's see how they perform. Okay. So now we got the results. Again, this took really a long time.
(05:32) And what immediately stands out is that GPT5 actually spent over a million tokens on this task, which is absolutely insane. Even GPT5 Mini is two times higher in token usage than Clot Sonet 4. And the performance for this task, as you can see, is actually the best for GPT5 with the score of 9 out of 10.
(05:52) Okay, so let's see the graphs for this task. And yeah, as you can see, GBT5 is again the best in terms of performance, but it's far not the best in terms of token cost and latency, while Claude Oppo actually completely failed. So, this is actually quite interesting that Claude Oppos failed so badly on the research task. So, let's see the overall grades.
(06:12) And the winner for this specific use case is actually Gemini 2.5 Flash. So Gemini 2.5 flash actually spent only 30,000 tokens which is nothing compared to GBT5 and was still able to produce a decent result and this makes sense because as I said before these models are actually using their own native search capabilities and as we know Google is probably the best today at search.
(06:37) Okay, so let's quickly also take a look at the reports. So the reports look something like this. Again, I'll leave a link to that other video in description where you can easily customize these agents for your own use case. And yeah, the flash model actually doesn't look that much different from any other model.
(06:51) It actually generated a much longer report which is kind of interesting. Okay. And the last use case is the newsletter generation agent. So this agent is actually based on the agent that we're going to release very soon. And that agent is essentially a reverse engineered clot code agent. So this a very comprehensive coding agent that contains all the same tools as cloud code to create to-do lists to read and write files to use bash and so for this use case essentially what we're going to do is we're going to send an image of a specific website to this
(07:21) agent like a reference from Figma that it needs to create. We're going to send all of the assets to this agent and the agent will have to generate an HTML with this newsletter. So this use case is definitely way more visual. So you guys can actually see the results that the agent produces.
(07:38) So let's run the model comparison again. And here are the results. So again, the first thing that comes up is that Claude Oppus spent for a single newsletter, which is of course probably much cheaper than performing the same process manually. I mean, for a real human, it would have taken multiple hours to do this. So I guess the cost for performing this process manually would still be probably like at least 0 to 00.
(08:02) while for Claude Opus, you know, it's only 7.5. And one more thing that stands out is that Gemini 2.5 Pro actually took more than 10 minutes to complete this task. So, let's now actually look at the generated results. And let's compare them side by side. So, this is the Claude Oppus web page and surprisingly it doesn't look that great.
(08:21) As you can see, the margins are kind of off. It's offc center and overall like it doesn't really match uh the reference. Okay, now let's take a look at cloth set. And surprisingly, Sonet actually looks better than Oppus. So, as you can see, set is properly aligned and even the website footer is looking really good. Okay, now let's take a look at Gemini 2.5 Flash. So, Gemini 2.
(08:44) 5 Flash seems like misaligned the images. Let's take a look at Gemini 2.5 Pro. So, for Gemini 2.5 Pro, it doesn't look too bad, but again, I definitely like the cloth on it much more. Now, let's take a look at GBT5. And this actually looks really good. So this is also one more thing that I learned about GPT5 is that it's really good at UIs.
(09:03) Not even like backends. For back ends, I still prefer Sonet, but for front ends, GPT5 for some reason is amazing. So the GPT5 generated page almost perfectly matches the reference. The only problem is that basically social media icons on the bottom are different sizes. And now let's take a look at GBT5 Mini. This actually also looks really good.
(09:23) So not a big difference. And surprisingly for GBT5 Mini, the icons on the bottom are also screwed up. So it's funny that both GPT5 and GPT5 Mini made the same mistake. And lastly, let's take a look at Grock. And yeah, I mean, yeah. So Grock failed miserably. So my overall impression is that GPT5 actually does suck. Yes.
(09:49) So if GPT5 was released like 6 to 9 months ago, I think it would have been a gamecher. But today, many other AI model providers have already moved on and they have already released models that are either comparable or much better than GBT5 in terms of performance. In terms of cost, yes, it's pretty impressive how cheap the GPT5 actually is today.
(10:09) But again, because it thinks for so long, it actually doesn't produce that big of a difference. Keep in mind, however, that we did test only the high GPT5 version uh in this video. So, what it seems like to me right now is that OpenAI is just cost cutting. If you consider that OpenAI has 700 million weekly active users on Chad GPT, but only 3% of them are paid, then GPT5 starts to make sense.
(10:32) Additionally, what it seems like is that the GPT5 that people had access to before is not the same model that we have access to today after its release. I've heard actually a lot of people in the space like Theo or the tall guy from Cursor refer to GPT5 as the smartest and the best coding model out there. And from what I've tested today, this is just not the case.
(10:52) It's pretty much the same level as cloud fornet on coding, but it just takes much longer. One thing I did note about GPT5, however, is that the hallucinations for this model are significantly reduced. It doesn't hallucinate as much and it can actually work really well for some of those analytics use cases like we saw with the first DB query agent.
(11:12) So basically the only time I would recommend today you use GBT5 is when you need to significantly lower the hallucinations or when you know exactly what you need to do. So also one more thing about GBT5 is that it's extremely good at following instructions. Code forceet for example can sometimes do things in its own way.
(11:30) You tell it to do one thing and then it does like five other different things. But GPT5 always does exactly what you tell it to do. And that can be a good thing but this can also be a bad thing. Like for example, sonet also does sometimes solve issues that I didn't even know about. While GBT5 like even if you tell it to completely delete your entire database, it's going to do that.
(11:50) While clotsson is probably going to propose to you do something else. And so again basically when I recommend using GBT5 is for analytics and rack use cases. Clot onet I recommend using for coding use cases and Gemini 2.5 pro I recommend using for any other general agentic use case. while Gemini 2.5 Flash I recommend using for any web browsing use cases.
(12:11) So yeah, that's my recommendation. Make sure to stay tuned for more super cool agentic builds that are coming soon, like for example, how to rebuild Cloud Code. And don't forget to subscribe.