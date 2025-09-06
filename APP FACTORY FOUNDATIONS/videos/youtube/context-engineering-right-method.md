# The RIGHT Method for Context Engineering (+3 Advanced Techniques)

- **Source**: YouTube - https://www.youtube.com/watch?v=BkfLE6gMmM4
- **Date**: Recent
- **Channel**: AI Engineering
- **Duration**: ~32 minutes
- **Key Topics**: Context engineering, AI agents, multi-agent systems, long horizon tasks, production reliability

## Key Points

- **Context engineering is NOT just providing the right information - it's providing the right information at the RIGHT TIME**
- Context must dynamically update as agents work - static files aren't enough
- Emerged now because we want agents to do complex, multi-step autonomous tasks (not just trip planners)
- Context from every step impacts all next steps - complexity multiplies
- Even with larger models, relevant context selection remains critical for accuracy, latency, and cost
- Long horizon agents are impossible without context engineering - reliability multiplies at each step

## The RIGHT Method (5-Step Framework)

### 1. **RETRIEVE** - Fetch Available Information
- Internal systems (Zendesk, Slack, Notion, project management)
- Web scraping and MCP servers  
- Previous agent memory and PRD files
- **Key**: Get all potentially relevant context

### 2. **INTEGRATE** - Add Context to Agent
- Chat history injection
- Agent state/internal context variables
- System prompt integration
- Vector database/RAG integration
- **Tip**: Put most relevant context at bottom of conversation (LLMs remember last info better)

### 3. **GENERATE** - Agent Response & Tool Calls
- Structured outputs and guardrails
- Tool execution with context-aware decisions
- **Critical**: Context impacts tool calls, not just responses
- **Key Difference**: Agents do tasks, chatbots just respond

### 4. **HIGHLIGHT** - Extract Relevant Context
- **Observability**: Track what agent actually sees (2 lines of code setup)
- **Summarize & Extract**: Use LLM to pull key moments and decisions 
- **Key Questions**: What made agent make mistakes? What is agent missing?
- Fine-tune prompts/tools based on context analysis

### 5. **TRANSFER** - Pass Context Forward
- Update internal systems with new information
- Send processed context to next agents
- Add to agent memory/scratch pad
- **Critical**: This step updates the first step - creates feedback loop

## Three Advanced Techniques

### 1. **Asynchronous Context Synchronization**
- **Tool**: Airbyte for data source syncing
- **Purpose**: Sync Slack messages, project data, customer tickets to vector DB
- **Key**: Real-time updates as team collaborates
- **Implementation**: MCP servers to query synchronized data
- **Result**: Agents have same context as human developers

### 2. **Self-Maintained Agentic Scratch Pad**
- **Purpose**: Agent updates its own working memory
- **Implementation**: Workflow files that define context management rules
- **Key**: Agent maintains relevant state throughout long processes
- **Integration**: Works with cursor rules and AI IDE workflows

### 3. **Multi-Agent Context Compression** 
- **Problem**: Cognition showed multi-agent systems fail due to information loss
- **Solution**: Custom communication flows between agents
- **Implementation**: Agents pass summaries and key decisions (not just messages)
- **Parameters**: Add "key_moments" and "summary" to inter-agent communication
- **Result**: Reliable multi-agent coordination without information loss

## Production Implementation Insights

### Real-World Context Requirements
- **Large Codebases**: Need access to Slack history, documentation, PRD files, and team decisions
- **Developer Collaboration**: Agent needs same context as human developer joining project
- **Complex Tasks**: Can't work from single PRD - need accumulated team knowledge
- **Multiple Repositories**: Context spans across repos, databases, and communication channels

### Critical Success Factors
- **Dynamic Context Evolution**: Context changes as agent progresses through task
- **Observability is Essential**: Must see what agent actually processes (not just inputs/outputs)  
- **Information Hierarchy**: Recent context more important than historical
- **Error Multiplication**: Each step compounds context-related mistakes
- **Tool-Context Integration**: Context directly impacts tool selection and parameters

### Technical Architecture
- **Airbyte**: Synchronize data sources (Slack, Notion, Zendesk, etc.)
- **Vector Databases**: AstraDB with embedded inference for searchable context
- **MCP Servers**: Query and access synchronized data from AI agents
- **Workflow Files**: Define context management rules for AI IDEs
- **Custom Communication**: Extended agent-to-agent parameters for context transfer

## Business Impact & Applications

### Use Cases That Require Context Engineering
- **Customer Support Agents**: Need ticket history and resolution patterns
- **Marketing Agents**: Require campaign history and performance data  
- **Lead Follow-up Agents**: Need CRM data and interaction history
- **Developer Agents**: Require codebase knowledge, team decisions, and documentation
- **Trip Planning Agents**: Need preferences, booking history, and real-time availability

### Why This Matters Now
- **Autonomous Complexity**: Tasks becoming exponentially more complex
- **Production Reliability**: Single mistakes can ruin entire multi-step processes  
- **Business Applications**: Moving from demos to production systems
- **Economic Value**: Context engineering enables truly valuable AI applications
- **Competitive Advantage**: Reliable long-horizon agents are significant differentiator

## Key Takeaways for Developers

1. **Context ≠ Static Files**: Dynamic, time-aware context management required
2. **Observability First**: Can't optimize what you can't see
3. **Tool-Context Integration**: Context drives tool selection and execution
4. **Multi-Step Thinking**: Each step affects all future steps
5. **Production Focus**: Design for reliability, not just functionality
6. **Framework Approach**: Systematic method beats ad-hoc context handling

## Transcript

(00:00) So context engineering seems to be the new trend and it's a replacement for prompting and vibe coding, right? Well, I personally don't think so. So I saw that video by Cole Med and many of his points definitely make sense. However, when I first heard about that trend, the first thing I thought was, aren't you guys already doing that? Like PRDs, all of these techniques, they're not new.

(00:29) you know, model context protocol is literally called like that because of that purpose. And so then I decided to dig deeper into this trend. And that's when I realized that what Andre Karpathy really meant in his post wasn't just using these techniques. It's not just providing the model with the right information, it's providing the right information for the next step.

(00:52) This last for the next step part is absolutely crucial and it's the hardest thing to get right. This is what everyone seems to have missed from this trend. And so in this video I will explain what's really so game-changing about context engineering, why this trend emerged specifically just now and even show you the right method we developed for context engineering and also the three secret context engineering techniques that you can use. right now. So, let's dive in.

(01:23) So, the best definition that I've ever heard of context engineering actually came from the 03D research model that I developed in my last video. And it goes as follows. It's the art and science of providing AI with just the right information at the right time. So again, not just the right information, but the right information at the right time.

(01:47) This means that the context must be dynamically updated as the agent keeps working. This is extremely difficult to get right as I said and I will show you how to do this later. But what you need to know right now is that it's more than just about providing static files to an agent. It's about actually making sure that the context evolves as the process unfolds.

(02:12) But why did it emerge just now and not a year earlier or later? And I actually thought a lot about this question and I think the answer is that now we are at a time where we actually want our agents to do stuff for us. We no longer just want to do stupid trip planners that literally generate random trips that we will never even use.

(02:36) We actually want these agents to build a trip specifically for our preferences and then not only plan it but to also actually book the entire thing including the flights, the hotels, the restaurants and literally do all of that autonomously. So we don't even have to think about it. And this is the most difficult thing to get right.

(02:58) The complexity of tasks that AI can perform autonomously is still growing exponentially and there seems to be no wall at all. So obviously as the complexity of tasks is increasing the amount of context to perform those tasks reliably is also going to be increasing. But the craziest part is that context from every step actually impacts the context required for all the next steps.

(03:21) So after each step, some context might become irrelevant while other context might suddenly become needed. For example, if your agent has already booked a flight, there's no reason for that agent to know the entire flight's schedule because it's just going to confuse it. And I know what you're thinking. As the models grow larger and smarter, they're just going to be able to pick the right context by themselves and we don't have to worry about the costs as much, right? But I don't think that this is how it's going to work in the future. I think it's always a good

(03:52) idea to pick only the most relevant context even when these models are 100 times cheaper and smarter than today because inevitably if you add some irrelevant information in the context, you're going to impact accuracy, latency, and costs. So just like today we have RAMs that can store terabytes of data.

(04:16) We still read and write from disks. I think even when we have LLMs with hundreds of millions of contacts, it's still going to make sense for us to provide only the most relevant information. Additionally, what's also crucial to understand is that the longer your task is, the more there are chances for an agent to make a mistake.

(04:35) You see, it's not just one to two% reliability difference between, you know, adding the most relevant context or not. This difference actually multiplies on every single step that the agent performs. So again, the more times that the agent calls a specific tool, the more times you have that 1 to 2% difference potentially create an error or an issue in the agent's process.

(05:02) And just one issue can ruin the entire process. So if the agent makes just one mistake, the entire trip could be ruined and then it would have to pretty much cancel everything and start over again. So the core advantage of context engineering and the reason why it evolved just now is because this is the only way to make so-called long horizon agents reliable.

(05:26) Without context engineering, these agents would simply never be reliable enough. So now what's the right method for context engineering? Well, I'm glad you asked. Keep in mind that it's not the only right method. Maybe you have your own framework, but this personally worked extremely well for us.

(05:44) So I'm excited to share it with you. It involves five steps and it's all about not just providing the right information, but again about providing the right information at the right time. So to illustrate it visually, I've created this diagram for you and I really wanted to make it as practical as possible.

(06:09) So basically inside the each step, you'll find the platforms or the specific techniques that you need to use at that step. So in the first retrieve step, what you need to do is simply fetch the available information for an agent that's relevant for the task. For example, this can be internal systems like if you're building a customer support agent, most likely you're going to use Zenesk. If you're building a developer agent like this is what I will show you later in this video.

(06:31) Most likely you're going to use Slack and maybe some other project management software like we for example use notion. Additionally, you can use of course web and MCP servers or you can use previous agents memory or those PRD files that there are plenty of tutorials on. After you fetch the most relevant information, the next step is to integrate it into the agents context window.

(06:57) And you can do this also using several techniques. The first one and the easiest one of course is to simply put it in a chat history or put it in a state of an agent like for example in the internal agent context like in OpenAI agents SDK.

(07:16) there is the context variable that allows you to slightly reduce the load from an agent because this context will automatically be used inside the tools. The next technique is of course to simply put it in a system prompt and the last one is to put it in a vector database. So basically add it to your agents knowledge using rack. After that of course you need to generate the response by an agent.

(07:41) So that is typically done automatically for you if you're using a framework or of course you can simply do it yourself by actually sending the request to an LLM. Additionally, of course, this involves using tools because context that you pass into an agent is essential for tools. You know, this is how agents actually provide value. They do not just output responses in chat. They actually complete certain tasks. And that's the difference between chatbots and agents.

(08:05) So you need to always be thinking not just about the context but how that context is going to impact the tool calls that the agent makes. By the way, one tip that I can give you at this stage is to always provide the most relevant context at the bottom of conversation history. It seems like LLMs because of their training remember the context that's closer to the last user message much better.

(08:29) And the last two techniques that you can use are structured outputs and guardrails. Both are also significantly impacted by the context you provided on the previous step. After the agent actually generated the response, the next step is to highlight. So highlight the most relevant context from that response. And you can also do this in several ways. The easiest way is to simply set up observability.

(08:55) Make sure to check out my last video with Adam Silverman. That video is incredibly valuable. Honestly, I watched it myself over two times and in that video, Adam showed everything you need to know about observability and how simple it is to get started. It literally requires only two lines of code and it gives you full visibility into what your agents are really doing.

(09:16) At the end, I even showed the demo and on that demo, you can see how easy it is to actually see all of the inputs and outputs by the tools and also everything that goes in and out of your agent. With this information, it's just so much easier to understand what the agent has actually seen that made it make a mistake or what the agent is actually missing.

(09:46) And these are actually the two questions that you need to constantly ask yourself on this highlight step is whenever agent makes an error or if something goes wrong, you need to ask yourself what has the agent seen in the context window that made it make this mistake. And without observability, you are probably simply not going to find out what this is, especially if you're using a framework.

(10:04) The next technique on this step is to summarize and extract. So summarization and extraction are crucial in context engineering because this is how you can actually extract only the most relevant information for the next step. You see you don't want to pass down again the entire context because it might just overwhelm the agent.

(10:28) So you need to ensure that only the most relevant details are passed to the next step. And you can do this by simply running another LLM and using a special prompt like can you please extract the key moments and decisions made during this conversation. Cognition actually showed this in their blog post where they argued that multi- aent systems don't work because agents typically don't pass all of the necessary information to the next agent.

(10:53) And then the results that two agents generate are not aligned and actually can't be used together. So in order to avoid that they're actually not using multi- aent approach. They're using a single agent approach and essentially before passing all of the information to another sub agent they use their own special fine-tuned model to extract all of these key moments and decisions.

(11:13) So I will actually show you at the end of this video how to do the exact same thing with multi- aent systems using our framework. There is actually a way in our framework where you can also do the exact same approach but still use a multi- aent system which is something that I have not seen in other frameworks.

(11:31) And the last way to highlight the most relevant information is to either fine-tune that agent or adjust things like system prompt or the tool inputs and how you process the data. This will also help the agent to pick out only the most relevant details. After that, the last step is to actually transfer that most relevant context to either the next agent or to the next step.

(11:57) So you can do this also by putting the information back where you actually retrieved it from. So updating the internal systems like for example if the agent answered the customer support request of course it makes sense to add this response on Zenesk so that the same agent can then use it later if the similar ticket is raised.

(12:17) Additionally you can of course send that information as I explained to other agents so they keep performing their tasks or you can add it to the memory or the scratch part of an agent which is another technique that I will show you at the end. However, this method doesn't end there.

(12:36) As you can see, there is a small arrow at the end of the transfer step and it actually goes back into the retrieve step. So, as I said on this last step, you need to think about how this information is later going to be used in the next iteration. And this is the key part. This is how you can actually keep this information relevant and up to date at all times.

(12:54) The last step in this method actually updates the first step as well. Okay. So, now that we understand what this method is, let me show you how to actually use it. And in this video, we're going to go over three advanced practical techniques that you have probably not seen anywhere else. The first technique is called a synchronous context synchronization.

(13:18) So, I know it sounds probably extremely confusing right now, but it will all make sense once we actually go through it. The next technique is called self-maintained agentic scratch pad. So this technique is a bit more known. I think you might have seen this somewhere possibly.

(13:35) But basically this means that the agent will be able to update its own scratch pad and then use it later in the process. And the last technique as I said is multi-agent context compression where essentially I'll show you how to make multi- aent systems reliable and avoid the problem that Cognition mentioned in their blog post with agents missing out on certain details. and then not being aligned on the task results.

(14:01) So again, all of these techniques, they do not just provide the agents with the right context. They ensure that this context stays relevant throughout the entire process. Now, let's actually jump in. Okay, so for the first technique, we're going to be using a platform called Airbite.

(14:21) Airbite is essentially a way for you to sync your data sources and synchronize the information between them. I will also show you how you can customize this to your own needs specifically for your use case and actually without this technique you can't even make agents reliable on large scale projects. So for example, I'm going to show you how to work with our project agency AI, right? Which is our SAS platform.

(14:46) And this platform has an absolutely massive codebase. And of course, we also have thousands of messages in our Slack, which our developers sent to each other while they were building this project. And without this key information, the agent simply is not going to be able to work and collaborate with other developers on this project. So you see, just generating a PRT can work just fine.

(15:10) If you're building like an example app that's probably never going to be used in production, but if you're building a real large scale project, then you can't even do it from a single PRD. There are just so many key moments and decisions that go into it, which is why you need to make sure that the agent has access to all of that information just like any other developer would.

(15:33) Okay, so let's try to create our first connection. It's actually pretty simple. And as you can see right here, we have a bunch of data sources that we can use. So, for example, if you're building like a marketing agent, you might use Mailchimp. So, the agent has access to your previous campaigns. If you are building like a lead follow-up agent, you might want to use HubSpot.

(15:52) If you're building like a customer support agent, then you might use Zenesk, right? So, again, the more customer support tickets you answer, the more information the agent is going to be able to have. So, let me find Slack and then let's connect our Slack account. Okay, after you connected your Slack account, we just need to fill out some settings.

(16:16) So, for example, in threads look back window, you can set how many days in the past the agent is going to look for the messages. And the start date is essentially the date from which the sync is going to start. So, I'm just going to set this at the end of the month because typically we do two week sprints.

(16:35) In the optional fields, you can also increase the number of concurrent workers if you have a lot of messages. And then here under the channel name filter, we need to just enter all of the channels that you want this agent to fetch. And by the way, this platform also has a very generous free quota and it's also fully open source, so you can even self-hosted later if you wish.

(16:52) Okay, after I entered all of the Slack channels, I'm just going to hit set up source. And now we need to define the destination. So for this video, we're going to be using Astra DB connection, which is a very convenient vector database. I find it super nice to work with because they have number one serverless options, meaning that you can get started completely for free and you also pay only for what you use and also they have integrated inference, meaning that you actually don't even have to create your own embeddings. You can actually create embeddings on their side. So let's also

(17:24) enter all of the parameters. So the chunk size is essentially the maximum number of tokens per one embedding that air bite is going to sync from slack. Uh fields to store as metadata is essentially all of the fields that air bite will save from slack to your vector store.

(17:44) So let's enter all of them right now. And then in the text fields to embed basically you need to enter all of the contextually relevant information like for example of course the message itself and also sometimes the channel ID might be relevant because this is what the agent is going to search for when curing your database.

(18:11) So let me also add text here and then also the channel name in the text splitter. You can also enter separators, which is essentially how Airbite is going to separate all of the different chunks. I don't think we need any separators here because typically Slack messages are self-contained and very rarely are they more than 4,96 tokens.

(18:31) Next, make sure to enter your open AIPI key. And then here, we need to enter all of our Astrad details. So, let's sign in on Astrad. It's actually pretty fast. And then here under databases, let's create a new database. So you can select whatever providing region you prefer. Just make sure you selected serverless vector database.

(18:52) And then we just need to wait a few minutes until our database is initialized. In the meantime, we can get our airbyte token, which you can find under settings in tokens. So here, just hit generate new token with the administrator user role. And then the token that you actually need to copy is this one at the bottom.

(19:15) So just copy that and put it back into air bite. Then after our database is initialized, we can go under overview and then also copy the API endp point. And finally we also need to create a key space. So a keyspace is essentially a specific segmentation of your data where for example you can have one collection for only the SAS development and another collection for ass development you know agents as a service. So I'm going to call this one SAS def as well.

(19:44) And then here you need to add also your own open AI key. There will be like a setup details here and then ARDB will be able to create your embeddings for you. So just make sure you select text embedding add a 002 model because this is what airbite expects.

(20:04) After you inserted all the details just hit create a collection and then you simply need to copy your collection name. So insert it into the ARDB collection and then for the keyace select default keyspace. So I think that's pretty much it for the setup on air byte site. So let's hit set up the destination. Okay. And now we also need to select the streams.

(20:32) So for the streams of your data, just remove channel members and also user streams because we don't really need those. And then hit next. And here you can also set the frequency of your syncs. So by default it's set to every 24 hours. This is going to depend on your integration. So for me, I think it's going to make sense to set it for every hour because in our Slack typically there are updates coming up quite frequently. Everything else you can set as it is and then simply hit finish and sync. Awesome.

(20:59) So now air bite is going to start syncing your data from Slack to Astradb and then you're going to start seeing some vectors appear here in your new collection. Okay. And while these messages are syncing, the next step is to of course connect to our vector database. For this, we're going to be using MCP servers. So let's jump into cursor.

(21:21) And now we just need to build a very simple MCP that will allow cursor to actually read our previous Slack messages. So for this, it makes sense for us to check the docs. Make sure you actually read the docs yourself before sending it to cursor because this can save you a lot of time.

(21:39) and let's just find like a code snippet or example that we can use to query our database. Okay, so here in the quick start I found a pretty good example. So let me just copy that. Let's add this into the prompt. Let's also copy the code. And for the MCP server itself, I'm going to be using fast MCP which is a very lightweight framework for creating MCP servers.

(22:00) And they also have the LLM's full documentation which you can also either just copy or insert into the cursor prompt or you can also index it right here in docs and then refer to it later. So indexing actually I think works better. So let's do this right now. And I'm also going to just copy the GitHub link just in case.

(22:19) Make sure to also list the specific tools that you want MCP to have. For this we just need two tools. List collections and query asterd. And also make sure to of course link the documentation page that we just added. Okay, so now let's hit send. Okay, so let's quickly check the code.

(22:39) So the code looks good, but I just want to return only the text from the document without too much extra information because it can actually confuse the agent. And also I think I forgot to specify which protocol I want to use. And for this I want to use stdio because it's easier to run with cursor. So, if you don't want to do this yourself, I'll leave the code for this MCP server and everything else in the description.

(23:09) By the way, a super cool tutorial on how to create MCP servers using our framework, which is hands down the easiest way to create MCP servers ever. We created a super convenient interface that just allows you to completely forget even about the MCP protocol itself and just create tools for your agents like you normally would. and then the framework converts it into an MCP server by itself.

(23:26) So I'll release this video probably somewhere next week. But in the meantime, make sure that you just tell the cursor manually to test the server before proceeding. Okay, perfect. So now, as you can see, the server is fully tested and it actually returns some of the information from our Slack channels. Amazing.

(23:43) So now we are ready to use this in cursor. You can also just tell cursor to generate a cloud desktop or a cursor configuration. Okay. So now we can just copy this MCP JSON file. Then we can go to cursor settings and then here we can just click add new MCP server.

(24:10) However, make sure you specify not just global Python but the Python inside your repo. Make sure to also create a virtual environment here. As you can see, cursor has already suggested me to do this. And then make sure to specify the full folder path. So, as I said, I'm going to leave all of this configuration down below. Okay. So, now, as you can see, we have two tools enabled.

(24:27) So, let's start a new chat and see what comes up. So, I'm going to use GBT4.1, and I'm just going to ask it to describe our GitHub app deployment project. So for now we also need to specify the collection and also which tool to use because right now cursor doesn't have the workflow file which is the next technique that we'll do.

(24:52) Okay, as you can see cursor now knows literally everything about this project. It's actually quite insane. It knows how the deployment process is initiated in our new feature. It knows when the app is connected. It knows what's the template repository that we're using. It knows the key capabilities like for example that we now create agencies using agencies swarm that we deploy them using E2B uh that you can see build logs and statuses.

(25:17) Additionally, it even knows the technical solutions we're using like for example that we're using capilot kit with a GUI events to support any framework. It also knows all of the limitations that we run into and also how developers collaborated and even the preview links that we're using. This is just insane.

(25:41) You know, this is how you actually make agents reliable on large scale production projects because all of this information is just so necessary for a real developer to build a production ready feature. You see, when there are multiple developers collaborating, there's no way that you can just create a PRD file out of nowhere and then make it actually work in a code base of this size. you actually need to ensure that the agent is aware of all of the previous decisions and what everyone is actually is doing in this team. So that's pretty much it for the setup with Airbite.

(26:10) And now I also want to actually connect a notion MCP server because this is where we connect our task. So I'm going to just do this right now super quick and then I'm going to show you how to combine all of this together using a special workflow file which is a second technique. So it's actually on cursor's MCP directory.

(26:29) So we can just hit add notion MCP and then click install. So let's now connect our workspace. Now as you can see our notion MCP is also working. Amazing. So now I want to use all of this together and try to develop a real production ready feature on our codebase. However, for this of course we need to create first a special rules file and this is the second technique.

(26:54) So the path where you create this workflow file is going to depend on which AI IDE or coding agent you use. For cursor it's under cursor. You need to create a new folder called rules. And then inside these rules you can add specific workflow files. So the workflow that I want to use for this agent is I want to make sure that it can check the tasks that are assigned to this agent.

(27:17) Like for example, we're going to try to develop this task where essentially we need to refactor some of the code in our codebase and move it into a separate repo. So essentially I want my agent to first be able to read this task, read all of the comments, read all of the supplying files that we have for this project and then based on this information actually start developing the app. So let's open the chat. Let's select cloud 4 with the max mode for this feature.

(27:44) And then basically all I need to do is just send it a task link and tell it to execute this task. And also don't forget to include the workflow file just in case. So first, as you can see, it reads the task itself. Then it reads the subpages in this task and also all of our internal documentation.

(28:09) So it has already pulled like several pages from our internal notion with exactly how we work on this project. It also pulled all of the latest comments from this task which as you can see there are also plenty of and then it pulls the Slack messages from the project channel. So this is as you can imagine a ton of relevant context that's absolutely necessary for this agent to perform the task. And this is exactly what a real developer would do on such a project.

(28:32) You would never just jump straight into the code. You would always first explore all the internal resources that we have, all of the documentation, of course, the Slack messages, and only then you would start coding. And now it actually creates a to-do list with all of the specific items that it needs to complete, like for example, all of the individual functions that need to be migrated. Okay, so now it's creating a new branch for this task.

(28:56) So now it's already on the to-do item number nine out of 17. It has already modified over 13 files and it just keeps going by itself without me even have to do anything. So let's wait until the process is fully completed and then we'll see the PR. As you can see, it created the PR with the changes that I requested.

(29:19) So yeah, there are definitely quite a few changes here. As you can see on the left, it changed over 23 files in just this repo, but there are actually more repos that it has modified. So yeah, that's that's pretty insane. Honestly, I'm still going to review the code, of course, before I merge it. I'm probably going to use another agent for that as well.

(29:37) But this definitely saved us hours of work or even more. And again, the context is maintained in notion so that in the future, our developers and future agents can refer to exactly what cursor has done in all of the next tasks. Okay. And the last example that I wanted to show you is how to actually make multi- aent systems reliable with context engineering.

(30:01) So again this relates back to the cognition's post where they said that you know multi- aent systems are not reliable because of the loss of information and they had to do this like sequential structure where each sub aent only has the key moments and decisions from the previous conversation.

(30:21) But I think this is actually quite possible to also implement with multi- aents because in our framework you can actually control the communication flows between all of the agents. So we have this cool feature custom communication flows which I have not seen anywhere else and this just allows you to have full control over how agents communicate with each other.

(30:39) So by default as you can see agents only send the message and also they can send specific files and additional instructions to each other. But what I'm going to show you in this last technique is how to actually make these agents also send the summary and key decisions that they have made during the process. So I'm going to name this file send message with context. And then here I already have this example.

(30:59) So let me just quickly walk you through it. So as you can see in our framework basically you can extend the parameters that the agents pass between one another. So to the default send message tool I'm just going to add two more parameters.

(31:17) one is going to be called key moments and the description for the agent is that this all of the crucial moments and decisions from the previous conversation and also the summary of all of them. So this is exactly like in cognition setup and then I'm just going to build like a very simple demo agency. So let's run this and let's now see how it works.

(31:38) And now as you can see we get success that the cost analysis tool was chosen correctly and both secrets are found when using this tool. Now let's try to remove the send message with context tool and let's just try to use a normal send message tool without the key decisions and next steps. So let's rerun this file. And now as you can see we get a failure because the expected secret was not found in the context of another agent.

(31:59) So it wasn't able to use the tool accordingly. So that's it for this video. These are the three examples that I wanted to show. All of the code will be down below in the description if you want to replicate these examples. if you want to play with them yourself or if you want to use our own workflow file template.

(32:17) So, thank you for watching and a lot more advanced and super cool tutorials are coming soon. So, don't forget to subscribe.