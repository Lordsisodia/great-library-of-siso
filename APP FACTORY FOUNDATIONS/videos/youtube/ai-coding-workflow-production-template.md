# My AI Coding Workflow That Actually Works in Production (Template Included!)

- **Source**: YouTube - https://www.youtube.com/watch?v=3myosgzKvZ0
- **Date**: Recent
- **Channel**: AI Coding Workflow
- **Duration**: ~44 minutes
- **Key Topics**: AI coding workflow, production systems, architecture planning, testing, Firebase

## Key Points

- 4 levels of AI coding autonomy: L0 (manual) → L1 (assisted) → L2 (monitored/VIP coding) → L3 (agentic)
- Main workflow: Architecture Planning → Types → Tests → Build Feature → Document Changes
- Use information dense keywords (create, update, delete, add, remove)
- Always think about what AI can/cannot see (context management)
- Use smallest AI model possible for each task
- Common pitfalls: connecting to production DB, accept-all trap, technical debt explosion

## Architecture Planning Steps

1. **PRD (Product Requirements Document)** - Well-defined requirements
2. **Project Structure** - Clean architecture from start
3. **ADR (Architecture Decision Records)** - Document why decisions were made
4. **Workflow Documentation** - Clear AI instructions

## Production Template Structure

Event-driven broker architecture with Firebase:
- Frontend: React with real-time updates
- Backend: Firebase Functions with triggered processing
- Database: Firestore with security rules
- Storage: Firebase Storage for file uploads

## Transcript

(00:00) So, you've tried AI coding. You've probably built a few projects, but in the end, it always comes down to the same result. Your code becomes an unmaintainable mess that not even the AI can understand itself. Everything always feels smooth at the beginning, but in the end, you always end up losing the control over the project because you simply have no idea of what's going on.

(00:25) And this is what causes these deleted production databases, massive security flaws, ugly AI generated UIs, or very minimal increases in productivity as shown by the recent studies. Instead of you running the project and shaping it into something unique that you actually want to use, the AI is running it for you and you have almost no control over what it's doing.

(00:52) And so in this video, I'm going to reveal my main secret workflow that allows me to generate reliable AI code that I can actually ship in production fast. Additionally, I will share with you my complete AI coding template that took me years to develop and that we are actually still using to this day on our own SAS product. In the end, I will also build an incredibly valuable marketing vertical AI solution that I've personally been planning to build from scratch for a very, very long time.

(01:16) And so if you watch this until the end, I guarantee that you're going to be able to build something like this yourself, even if you've never coded before. Let's dive right in. Okay, so first of all, quickly, why am I the one to teach you about all this? Well, first of all, I'm running a 20 people development team.

(01:34) Second, we are maintaining an open source agent framework. And third, we are also building a large scale SAS platform. So not only I'm running an AI agency myself, but we actually also ship something in production on a weekly basis. Okay. Now before we dive into my AI production coding workflow, we need to understand why so many people get this wrong.

(01:54) And in order to do that, we need to understand the four levels of autonomy in AI coding. So the first level of autonomy is L0. This is a fully manual workflow where humans are doing all the work. L1 is human assisted. This is what it used to be 3 4 years ago when we had a copilot in VS Code that could only do completions for you.

(02:19) And when we used to copy paste code from chat GPT L2 is human monitored. This is also known as VIP coding. And this is where AI now handles most of the tasks and humans are only watching AI for issues. This started around a year ago and this is still the primary stage of AI coding for many people today. And the latest stage is L3 human out of the loop.

(02:41) This is where AI handles everything from start to finish and humans only review the PRs. This is also known as agentic coding and Sam Alman and anthropics team have mentioned this multiple times. This is the final stage of AI coding that many forward thinking companies and individuals are trying to adopt today.

(02:59) What's important to note is that each stage of AI coding has different tools. So for example for vibe coding the primary tools are replet cursor loable or vinsurf and for agentic coding the primary tools today are cloud codeex or cursor background agents. So now why do so many people get this wrong? Well remember that unmaintainable mess that I mentioned at the beginning. Here's the reason it happens.

(03:24) People skip L0, L1, and L2 and jump straight into L3. They create like 20 cloud code sub aents or even worse they ask cloud code to create 20 sub aents for them. And of course in the end people like this manage to ship absolutely nothing. Another common problem is that anytime you start a new agent it inevitably loses some context from the previous tasks.

(03:49) All the key moments and decisions, reasoning and next steps are unavoidably going to be lost. So the next agent has almost no memory of what's going on. And so when you have multiple agents working on the same codebase at different times with almost no previous memory of what the agents did before them, they are all going to be doing completely different things.

(04:10) It's like trying to go from city A to cityb on different trains, but every single train takes you in a completely random direction. This is particularly common with backends. Backends, in my opinion, are not only harder, but also more important than front ends.

(04:31) Not to say that front ends are not important, but the primary functionality of your product is always going to be on the back end. And while there are frameworks that solidify the project structure for front ends like React and Nex.js, there's not a single unified project structure or an architecture that you can use for all backends. On the back end, you have way more different patterns and options. This is why Lovable started from front ends and not from backends.

(04:57) So for example, if latency and user experience are your primary concerns, you might want to follow a pattern like event driven. And if you want to scale to millions of users, maybe a pattern like microservices is going to be better. While if you're building an internal project, a simple monolith could be fine.

(05:13) So as you can see, there are way more options on the back end than on the front end. And this is exactly why the step one of my workflow is planning the architecture before any code gets written. I actually remember how David Tondre asked me who I'd hire first if I started my SAS all over again. And I said an architect because when you get architecture right from the start, something magical happens.

(05:34) You don't need hundreds of programmers or 24/7 coding to keep it working. It just does it. So what you really need to get this right is a cleanly architectured project from the very start. And this is exactly what I'm going to provide you with at the end of this video.

(05:51) Of course, this is not going to be the best architecture for all projects, but this is a great starting point and you can use it as a reference. So, typically I like to structure this in four files. The first file is of course a well-known PRD. The second file is the project structure. The third file is actually quite interesting and I haven't heard anyone cover this anywhere yet.

(06:14) And this file is called ADR, architecture decision records. ADRs are special files used by software architects to document all the important architectural decisions and why they were made. And the last file is the workflow itself, which is exactly what I'm explaining to you now, but clearly communicated to AI. So I'll show you how to create all these files at the end. But for now, let me present the rest of my workflow.

(06:39) So the next step after you plan the architecture is to actually create the types. This is also a very crucial step that many people skip. Type checking is actually one of the most valuable techniques. I think it's the second most valuable technique in AI coding.

(06:57) Types make your agents way more reliable because they add an anchor to the model. So if the agent has defined all the types for the feature correctly, including request types, response types, component, and database types, there's actually very little room for this agent to hallucinate and make a mistake. Almost all coding tools actually run linders after the agent runs a certain tool.

(07:16) So if there is a mistake, the agent is going to see it and this is exactly what increases the reliability so much. However, just this technique is not enough for you to go from L2 to L3 coding which is why the next step is to actually generate the tests. So this is hands down the most valuable technique in AI coding.

(07:35) People focus on all these like frameworks on top of other frameworks, but they forget simple development practices. And with AI, I think it's actually even more important. Most of the issues when you're coding with AI come from the fact that AI simply loses the necessary context in order to complete it reliably.

(07:54) And so if your feature is really complex, the conversation history for that feature is obviously going to grow very large. and then all of the coding assistants like cursor or cloud code typically removes some of the messages in the middle or in the beginning and so then AI forgets certain details and introduces a bug.

(08:12) However, if you tell AI to write a test first when it still has all of the necessary context for that feature, then later it literally can't even fool itself because it's going to run the test and it's going to see that something is not right, especially if you've previously also defined the types.

(08:30) And so when you define these three things first, the types, the tests, and the architecture, you essentially built a railroad for AI to complete this feature reliably. When you've done these three things, AI literally cannot go sideways. There is simply no way that AI can fail if these three things are correct. And so in a well architectured project often all I need to do for an agent to complete my feature is simply write the types and then explain what the feature does send it to the prompt.

(09:01) It's a very minimal prompt and then the AI literally just runs by itself until the feature is fully completed. So this is the next step which I'm sure many of you guys are aware of which is to build the feature. So I'm not going to spend a lot of time here. What's actually cool is that if you've architected your project well, you can actually run multiple agents in parallel.

(09:20) You just have to make sure that these agents are working on different parts of the codebase. So I'll show you again how to do this later. And the last step which is actually also quite crucial is to document the changes. As I said in my right method for context engineering video, it's not about the right context, it's about the right context at the right time.

(09:38) And so to ensure that the next agent has the necessary context for the next step, you need to ask the previous agent to document all of the key architectural decisions that it has made in the ADR document that I mentioned. Now, before we actually dive into the coding part of the video, let me also share a few of my best tips for AI coding that I'm sure you're also going to find extremely valuable and also some of these production disasters and how to avoid them.

(10:03) So, the first tip is to use information dense keywords. Information dense keywords are words like create, update, delete, add, remove and so on. These keywords are very distinct and they have a very clear meaning. With these keywords, you can make it much clearer for AI exactly what you want and how to do it.

(10:23) So take for example this prompt. Make the order total work better. It should handle discounts and add tax. This is how most people write their prompts in cursor. Now compare this to this prompt on the screen. Update. Then you include the file. Then you include all the functionalities and at the end you say add test with the file path.

(10:42) So which one do you think is going to be clearer for AI? Tip number two is to always think about what your AI can and cannot see. So not enough context or too much context again is the main problem why AI hallucinates. And in order to avoid that you need to constantly be thinking would I be able to complete this task with the context that I've given to AI? And if the answer is no, then obviously you need to change something.

(11:06) The same thing can apply for when you provide too much context. But honestly, most of the people don't provide enough context. Tip number three is to use the smallest AI model possible for a given task. So models can also either be too weak or too strong for a task.

(11:25) And in order for you to optimize efficiency, you need to use the smallest model because it's going to be the cheapest and the fastest. So for me personally, I use GPT 4.1 for quick edits, typically only with comment K. GBT5 for moderately complex features or for analyzing the code. As I said in my last video where I benchmarked all the models, GBD5 is extremely good at analytics tasks.

(11:43) I use Sonet for most of my other general coding workflows. So set is actually my preferred coding model today and Oppus only when I know that I can oneshot a feature using a background agent. So, Oppus can actually be even more cost effective than Sonnet if you know for a fact that it's going to be able to oneshot a feature.

(12:03) And the last tip is to keep up with the ecosystem. So, the new AI coding tools are coming out almost on a weekly basis. And I definitely recommend trying out as many of them as possible and seeing how you can integrate them into your workflow.

(12:20) So, now let me share some of the most common production pitfalls with some funny stories and how you can avoid them. So the first production disaster that has gotten a lot of attention is a VIP coding disaster with Replet. So Simon Sherwood who is a CEO of Saster was wipe coding and documenting his entire journey on Twitter and on day eight Replet actually deleted their entire production database and Saster is a pretty big company.

(12:44) So as Simon said this would cost a lot for Replet. However, later, luckily, they were able to recover the database even though Replet initially said that it's not possible to do this. So, how do you fix this? Well, you simply not connect an AI agent to your production database. It's that simple. You create what's called a staging environment.

(13:05) Staging environment is essentially the exactly same project on AWS or Google Cloud with the exact same setup as your production project. So instead of connecting to real user data, it's simply going to be using a new empty database. The next most common pitfall is the accept all trap. So today 20% of AI generated code still recommends non-existing libraries and only 55% of AI generated code passes basic security tests. So I think the reason for this is because AI is still lazy.

(13:36) It doesn't really understand that it's coding in production. It thinks this is a game or another benchmark because this is what they see in their training data. And the fix is actually also pretty simple. You simply use a modern text stack with simplified security rules like Superbase or Firebase.

(13:53) In Firebase, for example, there's a special on call function that handles all of the authentication for you. So with this function, there's literally no way that you can mess up authentication because it's all handled by Firebase. And the last common pitfall is the technical depth explosion. So although you can verify whether the feature is working correctly or not, it's much harder to verify if the EI has introduced any technical depth, especially if you're new to coding.

(14:19) A study at Stanford University of 100,000 developers found that they produce 30 to 40% more code on average. However, a significant portion of the code needs to be reworked, leading to an overall net gain in productivity of only 15 to 20%. So I think the reason for this is because it's very tempting to push the code as soon as the feature is completed.

(14:38) But the thing is is that as soon as you push some unfinished code, it becomes very tempting to push more unfinished code. And so a rule that I like to add to all my coding assistants is no broken windows. Essentially this means that whenever there is something that can be improved, it needs to be improved right away.

(14:56) And the last most common pitfall is using over complicated frameworks on top of other frameworks. So I've seen so many projects like cloudflow, agent OS, the BMAT method and so on. And all these projects do is they essentially just create prompts on top of other AI coding assistants like clot code or cursor.

(15:16) And these coding assistants already have a ton of prompts inside which actually makes it much harder to steer. But when you add more prrawns on top of those other prons, it becomes almost impossible for you to actually change the direction of the project. The AI is just going to design everything by itself again and you're going to have absolutely no control over the project in the end.

(15:40) Now, let's finally jump into the practical part of this video where we are going to be building a cold DM outreach system. Okay, so this is what we're going to be building today. Honestly, this is one of the most impressive things that I've ever vioded. And what it does is it creates personalized outreach videos with AI. So, basically what we need to do is we need to enter the name of the prospect.

(16:06) Then we need to enter the 11 laps voice ID, upload our pre-recorded video, which looks initially something like this. Hey, watermelon, my name is Areni from YouTube, and I just have a quick question for you. So, as you can see, basically this uh cold outreach video starts from Hey Watermelon. And what this system does is it replaces Hey Watermelon with the name of the prospect.

(16:29) So, it looks like I recorded this video specifically for them. So, all we need to do is just upload this video and then select the second uh when the greeting ends, which for this video is around 1.5. Then we need to hit upload and create a job. So, these videos are actually processed in the background. And then once the drop is completed, we're going to see a live status update right here. So now the new entry appears here.

(16:54) You can see this really cool animation. And then when it's finished again, this is all done in the background. So you can even like go away somewhere if you're processing like 20 videos and then come back. And then you hit the download link and you can see the result.

(17:13) Hey Christopher, my name is Arceni from YouTube and I just have a quick question for you. Would you like us to save your team more than 150 staff hours? So yeah, this is this is pretty insane, guys. I mean, this one of the coolest things that I've ever built, and I'm sure this is going to get us a lot of appointments. So now, let me show you how you can do something just like this or even better yourself. Okay, so here's the template for this project.

(17:36) You're going to be able to find this template in our school community and the link to join is going to be down below. So this template follows a very cool event-driven broker architecture. This is by far my favorite architecture that we are currently using on our own SAS product.

(17:56) So Firebase specifically allows for this very nice architecture where you can subscribe to the changes in a database and then essentially your front end is updated automatically. Additionally, this template contains all of the rules files for EI that you're going to need to build anything without writing a single line of code. Awesome. So now let's go ahead and set this up.

(18:15) So the first step is to create a new Firebase project. This should only take a few minutes. So simply enter uh your project name. I'm going to call mine DM outreach agent. You don't have to enable Gemini and Analytics and then hit create project. Next, let's enable Fire Store, which is the database that we're going to use.

(18:33) Also, by the way, what's really cool about Firebase is that it significantly improves the developer experience, which means that you can ship much faster. To deploy your app with Firebase, you only have to run one command. For this project, we're also going to need uh Firebase storage, which is also located under the build tab. So, for Firebase storage, if your project does require Firebase storage, you also need to upgrade your Firebase account.

(18:55) Firebase in general is extremely cheap. So you don't have to really worry about the cost and there is actually a no cost location for storage as well. Okay, now we are almost ready to get started. So the next step is going to be for us to create a web app.

(19:15) So simply hit register a web app, add your name, click register web app, and then you need to copy this Firebase config and insert it into the environment file. So open your project in cursor or in whichever ID you prefer and then on the front you'll find the enth.example file. So simply hit copy paste it back into the front name it and then simply insert these credentials. Awesome. Now we also have to do the same for back end.

(19:41) So for back end you need to go under the project settings and then here under service accounts you need to click create new private key. Then drop this key into the back folder and inside the end file in the back folder you just also have to enter your Firebase service account path. Perfect.

(20:01) So now we are done with the setup for this project. So the next step is going to be to generate the PRD for whichever product that you're building. So what's really cool about having a PRD and all of the task templates directly inside your repo is that you have full control over them.

(20:22) So, for example, when you use repos like Taskmaster, you don't really have the templates that they're using for their tasks, and so you have way less control over how your agents are scoping the tasks themselves. However, if all of your templates for all the AI rules are in the same repo, then you can simply adjust them for whichever workflow you prefer.

(20:42) So all you need to do in order to create a PRD is just to ask cursor to ask you questions until it has enough context to fill out this template and then at the end of course you also need to tag the PRD file itself. So for scoping I actually do prefer GPT5 high. My favorite model for general coding tasks is cloth onet but for scoping I do find that GPT5 is really good.

(21:06) So let's hit send and let's see what questions it comes up with. Okay. So now I'm going to answer all of these questions with Whisper Flow. The name for the product is DM Outreach Agent. The primary user for this is a marketing agency and jobs to be done are when I upload a video with the 11 laps voice ID. I want to get my personalized video output so I can send it to other prospects. Okay.

(21:31) Now I'm going to insert uh this whole thing. And then also for big projects what I like to do is I like to ask another follow-up question like do you have any other questions before you can scope a PRD.

(21:49) So as you can see GBT5 is thinking that there are still a lot of missing pieces and that's why for big projects I do like to ask another followup first. Perfect. So now it came up with more questions and typically the second round of questions is actually much better than the first one. As you can see this one is a lot more targeted towards what I want to build.

(22:07) like for example whether the greeting is always at the start of the video which is honestly like a very important question. So let me now also answer all of these and then we'll actually start building this. So now please draft this PRD. Perfect. Now GBT5 generated the PRD file according to my answers to these questions. So let's quickly review the file. Yes. And the PRD is actually looking really nice.

(22:34) So, as I said, GBT5 is great at analyzing the data and planning. So, the next step is to open the task template. So, you'll find the task template in a special tasks folder and then simply send it to GPT5 and tell it to break down this PRD into separate tasks. Again, feel free to adjust the task template as needed. Great.

(22:54) So, now, as you can see, around 10 15 minutes later, GBT 5 scoped a bunch of tasks for me and these tasks. so far are looking very very good. So these tasks are following our exact architecture as you can see they're referencing real files from the template and they are also even using the broker eventdriven architecture exactly like I requested where for example here there's a create greeting job function that simply creates a document in fire store and then this document is processed asynchronously with a triggered function that automatically maps to this

(23:28) collection in fire store. So yeah, as I said, GPT5 is just insane at planning. It actually planned everything extremely well. And honestly, I don't even think that I would be able to do a better job myself in just 15 minutes.

(23:47) The only thing that can be improved, I think, is that the types here actually don't contain the models themselves. So, it only listed like the names for the types that we're going to use in these functions, but it didn't actually list the properties inside those types. And as I said, types can act as guard rails for the models. This is why you have to understand the workflow yourself.

(24:07) And so next, I'm going to just ask it to also outline the models themselves and the properties for each request, response, and database type. Additionally, I also definitely recommend to tell GBT5 to check the security rules. So security rules is how you secure your database in Fire Store.

(24:28) Essentially, you can write like these special rules files that allow you to check whether the user should have access to a certain document in Fire Store or not. So, these things are actually quite important to understand when you're working with a certain text stack. And you definitely do want to remind EI to check those security rules. Great. Actually, it did scope the fire store and storage rules before. I just didn't notice that.

(24:45) So, this template does prompt AI to scope those rules as well to make sure that your apps are as secure as possible. But make sure to validate this. And if this task for scoping the rules is missing, make sure to tell cursor to also scope the security rules.

(25:04) Also, it's always a good idea to provide the relevant documentation at the scoping stage, which is something that I forgot to do. So, I'm just going to send uh the latest doc for 11 laps. And then I'm going to tell GBT5 to rescope the task. So, now I think all of the tasks look awesome. As you can see, they also, by the way, have dependencies on the top so you know which ones to send first.

(25:20) And the next step is to actually build out the feature. So I'm going to be using cursor for back end and cloud code for front end. But this workflow is actually fully AI tool independent. Meaning that you can use whichever tool you prefer. The only thing is that if you're using any other tool but cursor, you simply need to reference the agents MD file which references all of the other rules for AI agents.

(25:45) So let's launch cloud code and then all I need to do is just simply use an information dense keyword implement. Then I need to link the first front end task which is going to be off road protection and also I'm going to link the agents MD file. So let's kick off the clot code. And now while cloud code is forming we can launch other cursor agents in parallel. So let's open the new tab.

(26:10) I'm going to use cloud force on it for the implementation because it is by far my favorite implementation model up to date. And then I'm going to simply again use the information dense keyword implement and link one of the tasks. For the first cursor agent, let's add the API 11 lapse wrapper. And for the second cursor agent, let's let's add the fmp greeting replacement service.

(26:36) So let's kick off these two cursor agents. And actually I'm also going to create another cursor agent. So basically you can create parallel agents for any tasks that don't have any dependencies. And I think the the next task that doesn't have any dependencies is also the fire store and storage rules. Awesome. Now we have four agents working for us in parallel on this project.

(26:59) Can you just imagine how much of an increase in productivity you get by launching four agents in parallel? And all you need to do is simply check on them from time to time and see if they're going in a correct direction. This is, I think, how coding is going to evolve in the future.

(27:17) You're simply going to be supervising multiple agents at the same time. And this is where actually some of the programming skills can still be useful. It's not in actually writing the code. It's in being able to review the code and make sure that the AI simply hasn't hallucinated. Okay. So now after the first task with 11 laps is completed, I'm going to send the second task for the 11 laps service and also before that by the way I've included the API key in the environment variable.

(27:47) So I'm also just mention that to cursor. Okay. Now the second task is already completed with the ffmpeg service and the third task with fire store rules also seems to be completed. So now I'm also going to add an actual video into the test data folder so that cursor can actually test it with a real video. So again this is the next step after the types.

(28:09) Make sure that your AI actually wrote a good test. So you need to tell it to test it with some real data. I'm going to send another task to create the callable to the third cursor agent that already finished implementing the rules. In the meantime, you can also see that cloud code has already finished implementing the first front- end task.

(28:29) So now I'm going to send it the second front- end task for the upload form and storage. By the way, my favorite way of keeping track of changes when working with multiple agents is to simply use the git changes. So here on the g changes you can clearly see all of the files that all of the agents are currently working on.

(28:52) So here's the 11 labs API wrapper and also the 11laps servers and the ffmpeact service. In less than 10 minutes they have already modified over 40 different files. So let's check on cloud code. Awesome. And now as you can see our callable create a greeting job has already been created as well. So I'm going to send it the next task which is the actual batch processing job.

(29:12) And as you can see this task actually depends on all of the previous tasks. And as you can see the ffmpeg service is already tested with the real video as well. And the 11 laps service also seems to be tested. So also obviously when you're working with so many agents you can't review all of the changes.

(29:30) So what I recommend doing is actually reviewing only the tests. So as I said in my workflow, tests are crucial for AI because they provide the guardrails. If the test is correct, then there is very little chance that AI can actually screw up the functionality.

(29:49) Tests do not eliminate the possibility of all the bugs in your codebase, but they definitely verify that the feature is at least working. So I'm going to simply look at the integration test that the AI has created and I'm just going to check whether the EI is fooling me or whether it actually tests it with a real file.

(30:07) And by the way, if you don't know how to code, you can simply again use AI and just pass this file and ask if you know AI is actually checking with a real video or not. So the FFMP pack service seems to be correct. However, the 11 lap service I think is still using mog data and not actually running the real API. So I'm going to again use the information dense keywords and tell the agent who was working on the 11 labs API add and then integration tests.

(30:32) So the difference between unit tests and the integration test is that integration tests are actually calling the real APIs rather than just using the mock data. And I personally prefer to always run integration tests until the functionality is completed. And then after the functionality is completed you can replace the integration test with the unit test. Awesome.

(30:50) In the meantime, as you can see, cloud code finished working on the front end. So, let's actually see what it looks like. So, if you want to run front end, simply navigate into the front directory and then run yarn def. So, for the front end, we're using yarn. You'll find the instructions on how to install it in the readmi. So, let's open this link.

(31:07) Feel free to, by the way, modify this later. So, let's click sign up. And by the way, there's also a sign up with Google already available. However, I forgot to add it on Firebase console. So, if you want to add sign up with Google, go back to your project on Firebase. And then you need to simply click here, add additional providers, and click save. And we also need an email and password method. Awesome.

(31:32) Now, let's go back to our app. And now, let's try to sign in with Google. Okay, so it seems like we got an error with permissions, which is kind of strange. I think the reason we got an error with permissions is because right now in the fire store database we actually used the production mode which means that all of our documents are not going to be accessible on front end by default.

(31:58) So yeah for now I'm just going to set it to true here so that we're basically bypassing all of the security rules and then once we actually deploy it we're going to deploy these rules that cursor has created for us here. Okay. So, let's try to sign in again. And now we get this not so nice looking interface, but I guess this should work.

(32:16) So, as you can see, it has all of the fields that I requested, like the first name, the 11 laps, voice ID, the greeting, and second, and also the file for the video. Okay. So, now I'm just going to tell it uh to remove like all of these uh useless components at the bottom, and probably also to make the homepage a bit nicer looking. Okay, in the meantime, our 11 laps API seems to be tested.

(32:40) So, let's see if cursor has actually created the file. So, yeah, it seems like now it actually tests real 11 laps API and generates real files. However, it also seems like it added way too many different test cases for a lot of different names. And by the way, if you don't know how to code, test cases are actually really simple to read.

(32:58) All you need to do is simply read the name of the function and that typically describes quite clearly what the test case actually does. So here you can see test synthesize greeting loan name. So cursor is apparently testing loan names generation like Christopher. So I'm just going to tell it to simplify this test and leave only one test case for hey John. Nice.

(33:21) In the meantime, our front end is already looking much better as well. As you can see it's a bit more modern. And I also requested to use the dark mode. Yeah, this is really good. So yeah, it's honestly pretty crazy that we're running like five agents in parallel and the speed at which this project progresses.

(33:42) I mean, before for a real developer, this would have taken like a week to do by himself. While now I'm literally coding this and it's not even been like an hour and the project is almost fully completed, which is just insane. Awesome. And now, as you can see, we also get the save file from the 11 laps test. Hey, John. And it sounds exactly like me. Hey, John. Awesome.

(34:07) So, now just one more task left, which is to actually combine all of these services together into a unified function that processes the video in the background. So, the task is named triggered process greeting job. And as you can see, it depends on all of the previous tasks that we've already completed. So, this one is pretty complex. And let's see if cursor can actually oneshot this. What's really cool is that because it was planned by GPT5 beforehand that had all of the context about the product and all of the other files that had to be created inside the dependencies here, we already have all of the files that previously did not even exist. So even before these files were created, GBT5 has already

(34:46) referenced them in this final task. So this means that the final agent actually has all the necessary context from the previous agents even without completing those tasks by himself. Actually there is one more task left for front end which to actually check for the drop status after the video has been uploaded.

(35:06) So now I'm just going to send it again to cloud code while the cursor agent is working on the triggered function itself. So now cursor already tells me that it finished creating the triggered function. However, I actually can't even see the test. So as you can see you can prompt AI as much as you want but unfortunately today it still makes the mistakes and this is why you need to understand the workflow yourself.

(35:30) Even if you added the workflow into the prompts you still have to track that the cursor actually executes this workflow or cloud code executes this workflow reliably. Right? And so here all I need to do is just tell it to actually create an end toend integration test. And now as you can see we get a new indicator on the dashboard that shows any loading drops.

(35:48) But right now I think since we don't have the collection yet in fire store it simply returns an error. So we can also improve that quickly. We can just tell it to not show anything if there are no documents. So again I'm only checking the tests as you can see in the test integration end to end. Basically it uses the video that I added in the data and the hey John greeting that I generated before.

(36:08) However seems to be really lazy today. It actually I think didn't even run the test itself. So this is super important as well. Make sure you tell your AI to run the test. Do not run the test yourself because this essentially going to allow you to close the feedback loop.

(36:27) Meaning that you don't have to pass the errors back and forth between the terminal and your coding assistant. You simply tell an agent to run a test and then it's going to run it and if there are any issues it's going to see them and then it's just going to keep working until the test is passing. Okay, in the meantime let's refresh our UI and now as you can see we get a much better looking error message on front end.

(36:45) So now Claude tells me that it tested all of the functionality end to end. However, I can't actually see the final video. So I think it might be hallucinating again or it's probably skipping something. So this is a good way to test is like whether you got the final result in the end or not. So just simply ask it like where can I find the generated video.

(37:04) Okay. So definitely I think it hallucinated. It told me that I can go to emulators and find the video but unfortunately it's not there. So in these cases, there are some example tests that cursor can use as a reference. So here is an example of how to run a triggered functions test.

(37:21) So honestly, my app is a bit more complex than probably an app that you're going to build because it uses like these background jobs. And for them, we need to show cursor what the actual tests look like for these types of functions. So now I'm going to just tell it uh to use a proper integration triggered functions test following this example.

(37:40) Okay, so now I think the new test is actually looking much better. I'm going to try to improve the prompts so it doesn't fail at like real integration tests in the future. But again, the best way is to just ask it like where can I find the result? And if it's not there, then it's probably not testing something properly.

(37:58) So again, as you can see, I'm pretty much running into the same issue all over again is that clot tries to create some mock test files that actually don't test real logic. And yeah, that basically is quite useless. Like you need to tell clot to actually test real functions and ensure that it actually runs it with real API keys. Okay, so now finally I think it fixed the test and let's see what we got.

(38:24) So we got this download link. So let's see if it works. Hey Alex, my name is Nice. I can still hear like a bit of watermelon, but I mean, yeah, this is like extremely close. Hey, Alex, my name is from YouTube. And also, it seems like the audio leveling is not there yet. So, maybe this is something that I also need to work on as well.

(38:51) So, let me just tell it that uh the timing needs to be at like 1.5 seconds and also that the volume is too loud. Okay. And in the meantime, I'm going to actually deploy these functions and then I'm going to just uh tell cloud code to integrate them. So let's open another terminal. As I said, it's super simple to deploy in Firebase.

(39:12) So all we need to do to deploy these functions in Firebase is like literally just run Firebase. However, first actually I also want to remove all the example functions from the brokers folder. Okay, so let's see the updated version. Hey, Alex. My name is Reni from YouTube and I just have a quick question for you. Hey, Alex. And yeah, I mean it's it's pretty close.

(39:36) I guess what I also have to add to this app is a lip sync. There are a lot of lip sync models on replicate that allow you to essentially like sync the lips to the video. And yeah, I think after that it's going to be just insane. Now, basically, I think cloud finished removing the example functions. Let me also tell it to remove relevant affected tests. Okay.

(39:55) And now let's just run Firebase deploy. Also, before deploying, don't forget to replace your project ID in the Firebase RC file. And while it's deploying, what I also recommend you do is ask this agent who worked on the primary end to end function to document all of the key decisions in the ADR document inside the back folder.

(40:25) So again, this is crucial because at some point you're going to start another agent and it's going to forget all of the issues that this previous agent ran into and how it fixed them. So the prompt is pretty simple. Just don't forget to do this whenever there's like a significant change or whenever you solved like an issue that took you longer than it should. Then make sure you tell the agent to document this in the ADR document.

(40:43) And now, as you can see, it documented all of these key architectural decisions that we did. like for example that we are running Fireways emulators over mocked services which is one thing that it's been struggling with for a while. So again this way the future agents don't have to repeat the same mistakes they can just look at this document and then they're going to be able hopefully to do it much faster.

(41:09) Okay, in the meantime while our functions are almost deployed, I'm going to copy the name of the function and then I'm going to send it to cloud code and tell it to simply connect it. And now let me also tell it to connect to the collection where you can actually see the drop statuses and updates. Okay, so now cloud code tells me that it's fully connected. So let's test it out. So let's sign in into our DM outreach.

(41:38) As you can see, by the way, the sign in is automatic if we were already logged in previously, which is quite convenient. So let's insert a name. Let's upload the file. And now let's hit upload and create a job. So let's see if this works. So it also even added this cool upload bar. And now it says that the job has been created successfully. So I didn't see an automatic update, unfortunately. But if we reload the page. Hm.

(42:00) Yeah, I'm still not seeing it. So, let's check the console. Oh, yeah. So, basically the issue is that this query requires an index. So, in Firebase actually sometimes you need to create like an index for your database whenever there's like a complex query that filters the results based on different parameters.

(42:20) And so, if there is an issue like this, you essentially you'll see like this link in the console. And then all you need to do is just click on this link and then just hit save. And finally wait until it's completed. Okay. So now our index has been created. So let's reload the page again.

(42:39) And yeah, now as you can see we in fact see our first greeting job right here. So let's hit download. And we get the file. This is insane. Let's see. So it's not perfect. I guess I can play with the second. And I think this should fix it. Let's see again. And now, as you can see, we get a real time update down below. And it's actually really fast. As you saw right there, it was like only one or two seconds. That's insane.

(43:05) Okay, let's see this. Hey, Christopher, my name is Lareni from YouTube and and yeah, this is pretty insane. I mean, it's good as it is even right now, but I think there are even more improvements that I can make. Hey, Christopher, my name is from YouTube.

(43:24) Yeah, I mean, dude, like I wouldn't know if if this was an AI or me. Like, this sounds like I actually recorded it for them specifically. Hey, Christopher, my name is Reni from YouTube and I just have a quick question for you. Would you like us to save your team more than 150 staff hours every single month? Yeah, that's insane.

(43:44) Okay, so now the last step is just also tell cloud code to document all of the architectural decisions so that again future agents don't make any mistakes and then I guess I'm going to come back to this uh and improve it even more. So just watch out for those DMs in the future from me on Twitter or LinkedIn.

(44:01) And now that these ADRs are documented, we don't have to think about AI losing the context. So again, next time when I come back to improve this system even further, which I'll definitely do very soon, the next agent is going to remember all of these key moments, decisions, and it's going to be able to work seamlessly on the same code base.

(44:21) So yeah, this is how you build real production maintainable systems. I'm going to be improving it even further in the future. And don't forget to subscribe.

## Related Components

- PRD Templates
- Architecture Planning Templates  
- Testing Framework Setup
- Firebase Integration Patterns
- Multi-Agent Workflow Management