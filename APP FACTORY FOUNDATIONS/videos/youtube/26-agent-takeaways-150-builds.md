# 26 Key Takeaways from Building 150+ Agents in 9 Months

- **Source**: YouTube - https://www.youtube.com/watch?v=jmeGqDu4tPU
- **Date**: Recent
- **Channel**: Agent Development Agency
- **Duration**: ~30 minutes
- **Key Topics**: Agent development lessons, business implementation, production deployment, agency experience

## Experience Base & Authority

- **150+ AI agents built** in 9 months under agents-as-a-service model
- **Real-world business implementations** across multiple industries
- **Hard-earned lessons** from dissatisfied clients, costly mistakes, and iterative improvements
- **Agency-scale insights** on what works vs what fails in production

## The 26 Critical Takeaways

### 1. **AI Agents ≠ Your Employees**
**Reality**: Agents have less autonomy than employees, more than automations
- **Agents**: Dynamic execution, require training on exact instructions
- **Employees**: Can learn from SOPs and adapt through trial/error
- **Thinking Framework**: 1 agent = 1 SOP, 1 employee = 5+ SOPs
- **Architecture**: Think in terms of processes, not roles

### 2. **Start from Well-Documented Processes** 
**Foundation**: SOPs (Standard Operating Procedures) are agent training gold
- **Advantage**: Skip data collection, reduce client questions
- **Source**: Existing onboarding materials and process documentation
- **Result**: Significantly simpler agent training and higher reliability
- **Priority**: Find documented processes first, then build agents

### 3. **Business Owners Will Never Build Their Own Agents**
**Market Reality**: Demand for AI agent developers will spike, not decrease
- **No-Code Parallel**: Created no-code developers, didn't eliminate developers
- **Core Challenge**: Not building agents, but determining which agents to build
- **Opportunity**: AI agent developer role will become more valuable
- **Business Model**: Consulting and expertise will remain essential

### 4. **Business Owners Don't Know Which Agents They Need**
**Discovery Process**: 50% of initial client ideas aren't the most valuable
- **Method**: Map customer journeys step-by-step on Figma
- **Focus**: Find automation opportunities within customer experience
- **Value**: Easier-to-build solutions often provide more business value
- **Approach**: Use client ideas as feedback, not requirements

### 5. **You Don't Need 20+ Agents**
**Simplicity Principle**: Start with minimum viable agent system
- **Problems**: Complexity, maintenance difficulty, debugging challenges
- **Cost**: More agents = higher operational costs and slower responses
- **Strategy**: Start with 1 agent, fine-tune completely before adding more
- **Deployment**: Test and validate each agent before system expansion

### 6. **Data + Actions = Maximum Impact**
**Integration Principle**: Knowledge alone isn't enough
- **GIGO Rule**: Garbage in, garbage out still applies
- **Breakthrough**: Combining data with relevant actions delivers exponential results
- **Example**: Facebook marketing knowledge + Facebook API control
- **Capability**: Agents provide suggestions AND execute improvements
- **Sources**: Scrape both internal and external data sources

### 7. **Prompt Engineering is an Art**
**Professional Skill**: Real job that companies don't yet recognize
- **Evolution**: More important as models get smarter and run longer
- **Approach**: Write prompts like blog posts or essays
- **Tips**:
  - **Examples**: One example worth 1000 words
  - **Order Matters**: Most important content at end of prompt
  - **Iterate**: Test constantly, adjust based on performance
- **Impact**: Simple rearrangement can transform unreliable to consistent

### 8. **Integrations = Functionality**
**User Experience**: Integration convenience determines adoption
- **Reality**: Powerful agent in wrong system = no value
- **Strategy**: Integrate into systems employees already use daily
- **Example**: Customer support agent must work in existing Zendesk
- **Focus**: Meet users where they are, don't force new workflows

### 9. **Agent Reliability Has Been Solved**
**Solution**: Pydantic data validation eliminates agent failures
- **Method**: Validate all agent inputs and outputs with Pydantic
- **Result**: Agent literally cannot execute harmful actions
- **Developer Responsibility**: Unreliable agent = developer problem
- **Implementation**: Check Jason Leo's "Pydantic is All You Need" approach
- **Libraries**: Instructor library provides practical implementation

### 10. **Tools Are the Most Important Component**
**Value Generation**: 70% of agency work goes into building actions
- **Components**: Instructions + Knowledge + Actions
- **Differentiation**: Chatbots provide responses, agents execute tasks
- **Value**: Agents must DO things, not just suggest things
- **Architecture**: Well-structured tools enable any use case
- **Priority**: Actions over responses for real business value

### 11. **No More Than 4-6 Tools Per Agent**
**Performance Rule**: Too many tools cause hallucination and confusion
- **Limit**: 4-6 tools maximum per agent (depending on complexity)
- **Models**: GPT-4o, Claude 3.5 confirmed limitations
- **Symptoms**: Agent confusion about tool selection and sequencing
- **Solution**: Split into multiple agents when hallucination occurs
- **Architecture**: Simplicity ensures reliability

### 12. **Model Costs Don't Matter**
**Business Reality**: ROI trumps model pricing concerns
- **Focus**: Right use cases provide tremendous ROI vs manual processes
- **Example**: $300/3 days manual → $1-2/20 minutes automated
- **Strategy**: Focus on value creation, not cost optimization
- **Economics**: Development costs matter more than operational costs

### 13. **Clients Don't Care Which Model You Use**
**Business Focus**: Value delivery over technical preferences
- **Reality**: Never used open source models in agency
- **Priority**: Results and policy compliance over model selection
- **Privacy**: Azure OpenAI for strict data privacy requirements
- **Choice**: OpenAI for developer experience and API convenience
- **Business**: Save development time = more client value

### 14. **Don't Automate Until Value is Established**
**Risk Management**: Prove process value before automation investment
- **Problem**: Automating non-existent or unproven businesses
- **Strategy**: Hire manual workers first (Upwork), establish process value
- **Costs**: Development investment vs uncertain process value
- **Validation**: Confirm process works and provides value before automating
- **Timeline**: Manual validation → documented process → automation

### 15. **Think ROI, Not Use Cases**
**Business Formula**: (Rate × Hours - Operational Costs) ÷ Development Costs
- **Components**:
  - **Rate × Hours**: Employee cost × total process time
  - **Operational**: Model + server costs (usually negligible)
  - **Development**: Solution building investment
- **Example**: $50/hour × 10 hours/week ÷ $5000 development = 5.6x ROI annually
- **Focus**: Highest value processes, not interesting use cases

### 16. **Agent Development is Iterative**
**Testing Approach**: Try multiple architectures, compare performance
- **Kaggle Lesson**: Most tested solutions win, not most knowledge
- **Questions**: Tool count, agent count determined through experimentation
- **Method**: Build variations, test side-by-side performance
- **Experience**: Pattern recognition develops over time
- **Process**: Confused performance = test more variations

### 17. **Use Divide and Conquer Approach**
**Delivery Strategy**: Break complex problems into manageable components
- **Benefits**: Incremental delivery, early validation, reduced risk
- **Method**: Build smallest working agent first, validate, then expand
- **Business**: Automate by departments first before cross-department
- **Integration**: Combine agents within departments for more power
- **Client**: Confirm each component works before building full system

### 18. **Evals Matter for Big Companies Only**
**Evaluation Metrics**: Track KPIs and performance over time
- **Enterprise Value**: Continuous improvement, competitive elimination
- **Network Effect**: Previous solutions improve with new clients
- **Future**: Enable agent self-improvement capabilities
- **SMB Reality**: Don't need evals due to low request volume
- **Performance**: 80% achievable without evals, last 20% needs them
- **Scale**: Large enterprise = evals from start, SMB = later

### 19. **Two Types of Agents: Agents vs Workflows**
**Architecture Options**: Fully agentic vs structured agentic workflows
- **Pure Agents**: Fully autonomous decision-making throughout
- **Agentic Workflows**: Predetermined steps, but steps themselves are agentic
- **Example**: Lead research with fixed Google search sequence, dynamic content
- **Tools**: CrewAI (workflows) vs custom frameworks (pure agents)
- **Use Cases**: Some processes need exact step sequences with smart execution
- **Future**: Adding workflow features to pure agent frameworks

### 20. **Agents Need Feedback Adaptability**
**Environment Interaction**: Must analyze results of their actions
- **Problem**: Agents modify environment but can't see impact
- **Solution**: Add analysis tools, not just modification tools
- **Example**: Database update + database read tools for verification
- **Design**: Every action should have corresponding verification capability
- **Learning**: Agents adapt based on environmental feedback loops

### 21. **Don't Build Around Limitations**
**Future-Proofing**: Build assuming models will improve
- **Mistake**: Complex systems to avoid 2023 context limits became obsolete in 2 months
- **Evolution**: OpenAI releasing operator, deep research, file search capabilities
- **Casualties**: Chat-with-PDF apps became obsolete overnight
- **Warning**: Don't build obvious use cases OpenAI might develop (like software development)
- **Strategy**: Build for future model capabilities, not current limitations

### 22. **Deploying Agents > Building Agents**
**Reality Check**: 2-3 days to build, 3+ days to deploy and integrate
- **Challenge**: Integration into client processes takes longer than development
- **Solution**: Custom platform for deployment flexibility
- **Problem**: Existing platforms built before agents, lack production flexibility
- **Business**: Deployment bottleneck limits agency scalability
- **Platform**: Need agent-specific deployment infrastructure

### 23. **Waterfall Projects Don't Work**
**Project Management**: Subscription model beats fixed-scope projects
- **Reality**: Agentic projects too agile for 3-month scoping
- **Evolution**: Discover new opportunities during development
- **Approach**: Partner, not outsource vendor relationship
- **Goal**: Automate business, not just build agent
- **Structure**: Agile service agreements with room for optimization

### 24. **Human-in-Loop for Mission-Critical Agents**
**Risk Management**: Some agents need approval steps initially
- **Criteria**: Low margin for error, high-cost mistakes
- **Example**: $100k marketing campaign approval before execution
- **Process**: Review in Notion, approve, then execute
- **Evolution**: Remove human step once agent fully fine-tuned
- **Safety**: Prevention better than expensive mistake recovery

### 25. **2025 = Year of Vertical AI Agents**
**Market Evolution**: Specialized agents for specific industries/use cases
- **Benefits**: Higher pricing, easier scaling, deeper customer knowledge
- **Strategy**: Don't start vertical immediately
- **Process**: Build horizontal agents first, identify patterns, then verticalize
- **B2B SaaS Parallel**: Vertical software commands premium pricing
- **Timeline**: Horizontal experience → pattern recognition → vertical product

### 26. **Agents Don't Replace People, They Enable Scale**
**Business Impact**: Help businesses think bigger, not smaller
- **Reality**: Never seen immediate employee firing after automation
- **Results**: Higher revenues, more profits, employee focus on high-value work
- **Employee**: Focus on tasks they truly enjoy
- **Vision**: Age of abundance and prosperity through human-AI collaboration
- **Philosophy**: Humanity finds better things to do with increased efficiency

## Business Implementation Framework

### **Phase 1: Foundation (Months 1-2)**
1. **Process Discovery**: Map customer journeys, identify SOPs
2. **ROI Analysis**: Calculate value potential using formula
3. **Simple Start**: 1 agent, well-documented process, 4-6 tools max
4. **Integration**: Deploy in existing systems employees use

### **Phase 2: Validation (Months 3-4)**
1. **Test & Iterate**: Multiple architectures, prompt engineering
2. **Human-in-Loop**: Safety measures for mission-critical processes  
3. **Performance**: Achieve 80% performance before expansion
4. **Department Focus**: Complete one department before moving

### **Phase 3: Scaling (Months 6+)**
1. **Multi-Agent**: Add agents incrementally after validation
2. **Enterprise Evals**: Implement for high-volume clients
3. **Vertical Patterns**: Identify commonalities for productization
4. **Subscription Model**: Agile partnership vs fixed-scope projects

## Critical Success Principles

### **Technical Excellence**
- **Pydantic Validation**: Eliminate agent reliability issues
- **Tool Architecture**: 70% effort on actions, not instructions
- **Prompt Engineering**: Treat as professional art form
- **Model Agnostic**: Focus on value delivery over model selection

### **Business Strategy**  
- **ROI Focus**: Calculate and prioritize by return on investment
- **Process First**: Document and validate before automating
- **Integration Priority**: Meet users in their existing workflows
- **Incremental Delivery**: Divide and conquer complex problems

### **Client Management**
- **Consultative Approach**: Guide clients to highest-value solutions
- **Subscription Model**: Agile partnership over fixed-scope contracts
- **Safety Measures**: Human-in-loop for mission-critical processes
- **Scale Enablement**: Help businesses grow, not shrink workforce

## Industry Evolution Predictions

### **Near-term (2025)**
- **Vertical AI Agents**: Specialized industry solutions dominant
- **Developer Demand**: AI agent developers increasingly valuable
- **Platform Evolution**: Better deployment and integration tools
- **Human-AI Collaboration**: Complementary rather than replacement

### **Technology Trends**
- **Model Improvements**: Build for future capabilities, not current limits
- **OpenAI Capabilities**: Avoid obvious use cases they might build
- **Context Windows**: Large contexts available but relevance still matters
- **Self-Improvement**: Evals enable agent self-optimization

### **Business Model Shifts**
- **Consulting Integration**: Technical + business expertise required
- **Subscription Services**: Agile agent development partnerships
- **Process Automation**: Documented SOPs become agent training data
- **ROI-Driven Decisions**: Value calculation over technology fascination

## Transcript

(00:00) we built over 150 AI agents under the new agents as a service model in the last 9 months in this video I'll share with you 26 key takeaways that we had to learn the hard way that costed us a lot of dissatisfied clients time and money so you don't have to repeat our mistakes let's dive in key takeaway number one AI agents are not your employees today everyone still seems to call AI agents either automations or your own employees and unfortunately agents are neither of those things you see the difference between agents and automations is that in automations every single step is

(00:45) hardcoded for you which means that you know the exact steps and the sequence of those steps in advance while the difference between agents and employees is that agents have less autonomy than employees meaning that typical you need way more agents than you need employees agents require training on the exact instructions in order to perform a process manually they can just take a look at your sop and then go out into the world and Learn by themselves using trial and error unfortunately not yet so instead of thinking about agents in

(01:22) terms of roles I believe you should actually be thinking about agents in terms of Sops typically One agent can handle one standard operating procedure well while one employee typically handles five or more Sops which brings us to our key takeaway number two start from well documented processes Sops are standard operating procedures these are the processes that employees perform in a given business and in a good business these processes are typically always well documented so by finding these well documented processes first you can make training an agent significantly simpler

(02:08) instead of collecting all of the data manually instead of asking tons of questions from your client you can simply take that sop and most likely it will have everything you need in order to train this agent to reliably perform this process so find these onboarding materials find Sops first and then go from there key takeaway number three business owners will never build their own agents I believe that even when we have agents that build other Agents from a single prompt business owners will still never

(02:44) build their own agents just like no code tools promised us the end of software developers but instead simply started a wave of no code developers and just like automation platforms promised us the end of backand Engineers but instead started a wave of automation Engineers a agent platforms and Frameworks that can build other agents will only Spike the demand for AI agent developers as open Ai and other labs keep releasing new agend capabilities like the operator and the Deep research agents it's not building agents that will be the hardest it's determining which agents to build which

(03:26) is exactly where a agent developers come in so don't worry about agents building other agents business owners will always prefer to entrust this to someone who knows how to work with EI agents best even when the process becomes extremely streamlined which brings us to key takeaway number four business owners have no idea which agents they need a lot of our clients when they start their subscriptions come to us with some ideas for which agents they want to build but in around 50% of of cases this is by far

(04:02) not the most valuable agents that we can build for that business which is why Consulting is a huge part of our service so to determine which best agents to build we typically prefer to start from a customer Journey we ask our clients to help us map out their customer Journeys on figma with us step by step and this gives us a great idea on which specific parts of that customer Journey we can automate so then if we see that there is a potential opportunity there we can dive deeper into that specific part of the process and then maybe even map out that process by itself and by doing this

(04:43) typically you can find opportunities that are not only easier to build but also bring significantly more value to that business so don't listen to your clients if they have some ideas for which agents they want to build use it as feedback don't assume that this is the best possible idea that you can integrate into their business which brings us to takeaway number five you don't need 20 plus agents you see right now on YouTube and in our community people seem to be building as many agents as possible and what this typically does is it only makes your systems more complex by adding so many

(05:24) agents in a single system you simply make it harder to maintain it becomes increasing more complex to debug the system and to find potential issues additionally it increases cost and the amount of time it takes for your agents to provide a response so start with as few agents as possible preferably start even from one smallest agent that you can deliver to your client as fast as possible and then once this agent is fully fine-tuned once you actually had a chance to deploy this agent and once the client has tested it then you can

(06:01) proceed to adding more and more agents as needed which brings us to key takeaway number six data driv decisions but data with actions deliver results you see there is a say in data science called gigo which stands for garbage in garbage out and this still holds true for eii agents if you provide your agents with trash inputs they're going to produce trash outputs however what we discovered just recently is is that the biggest impact doesn't come just from adding data to an agent it comes from combining that data with relevant actions together by combining knowledge

(06:42) like for example how to create effective Facebook marketing campaigns with actions that allow this agent to control Facebook marketing API you can achieve significantly higher results than by using either data or actions Alone by combining knowledge and data the agent is not just executing tasks it can actually provide you with the suggestions on what exactly you could improve and how to perform that process best so make sure to scrape both internal and external sources this will significantly increase your agent's performance which brings us to key

(07:20) takeaway number seven prompt engineering is an art honestly prompt engineering is already a real job even though many companies still don't recognize it as one and as these models become larger and smarter and are evolving to run for 10 minutes or more prompt engineering is becoming more important than ever before so right now prompt engineering is already like an art you have to write your prompts like you would be writing blog posts or essays you have to think carefully about every single word you put in and add prompt because again as the models get smarter prompt

(08:00) engineering will become more and more important so here are some tips for writing effective prompts tip number one provide examples make sure you provide enough examples to an agent because often one example is worth a th000 words number two order matters what we found is that actually the order of your sentences or paragraphs in your prompt makes a large difference from our agent developers experience we often had agents where simply by rearranging the prompt the performance of the agent went from completely unreliable to consistent

(08:40) and reasonable make sure that the most important parts of your prompt are at the end of instructions not at the beginning because large language models remember information that's closer to the latest message better and the last tip is iterate and test constantly often times s the only way you can determine whether your prompt works or not is by iterating and testing it so do not modify your prompt without testing how it affects the performance of the agent make sure you run your agent as often as possible and then adjust the prompt accordingly which brings us to key takeaway number eight Integrations are

(09:21) just as important as functionality often times we tend to over focus on the agent capabilities however integration which is where the agent is working are often times even more important because if it's not convenient for your users to use an agent it doesn't matter how powerful your agent is it's still not going to be able to provide any value so again make sure that you integrate your agents into the same exact systems that your employees use daily if you are building a customer support agent and your client currently uses zenes the agent must also work in zenes as well

(10:00) which brings us to key takeaway number nine agent reliability has been solved so today a lot of startups are trying to solve agent reliability but in our agency if the agent is not reliable it's not the agent problem it's the developer problem and the person who is responsible for solving agent reliability is named Jason Leo back in 20203 he released this legendary video called penic is all you need and he also recently L posted an update and guess what penic is still all you need essentially he figured out that you can use penic a data validation library to

(10:39) validate all agent inputs and outputs which means that if you added all the necessary validation Logic the agent literally cannot screw anything up it cannot take any action that would cause any major consequences because it should be prevented by the developer who was developing this agent so make sure to check out this video make sure to check out how it's implemented in his Library instructor and in my framework because honestly with penic you can even now build agents for literally any use case which brings us to key takeway number 10 tools are the most important component

(11:17) when it comes to building AI agents there are three most important components which are instructions knowledge and actions and around 70% of work in our agency goes in to building actions which are the tools why is that well because tools is how agents provide value you see with standard chatbots or llms the value is generated through responses while agents generate value through actions agents must execute tasks they shouldn't just tell you what you need to do or simply provide a response to your cury agents should

(11:54) actually do that thing for you which is why actions are the most important component if you know how to build and structure your tools well you can literally build agents for any use case which brings us to key takeaway number 11 no more than four to six tools per agent this is our rule right now we are not adding more than four to six tools per agent depending on their complexity this has proven to work the best in our e agency although of course it does depend on the complexity of your tools right now with at least GPT 40 and Claw 3.5 and latest GPT models you can't add

(12:35) more than six tools because the agent simply starts to hallucinate the agent starts to confuse which tools to use or what's the proper sequence for using these tools this is a really good rule of thumb if the agent starts to hallucinate you need to split this agent into multiple agents key takeaway number 12 model costs don't matter recently DPS made a lot of noise and it's certainly impressive what they have been able to achieve in terms of costs but honestly we stopped carrying about model costs a long time ago if your use case makes

(13:11) sense you will almost always make a tremendous Roi from using an AI agent compared to performing the same process manually if you simply focus on the right thing then you no longer have to worry about costs so check out my previous video with five agent case studies where for example for one agent the process for filing questioners was reduced from $300 and 3 days manually to $1 to $2 and about 20 minutes which brings us to key takeaway number 13 clients don't care about which model you use many people are typically surprised

(13:51) when they hear that we never used an open source model in our agency businesses don't really care about which model you use as long as your use case makes sense if you can provide them with value without violating their customer policies then they don't really care about what model provides that value so in case if our customers do have these strict data privacy policies with their own clients what we do is we simply use Azure openi which essentially runs openi models in your own private Azure instance without even sharing data with open AI itself openi is still our

(14:33) provider of choice because of their developer experience it saves us a significant amount of time on developing agents because of how convenient it is to work with their API which does significantly simplify the agent creation process which brings us to key takeaway number 14 don't automate until value has been established although we do work primarily with existing businesses sometimes we get people who come to us and they want to automate a business that doesn't even exist so they want to build a process not from

(15:09) establishing this process manually first and then automating it but from guessing that automating this process will make more value so in my opinion this is extremely risky because you don't even know if the process is going to work at all and it's going to require a significant investment typically it's the development costs that you need to worry about rather than the model costs so the development cost is where you know the solution might not make sense which is why you need to First establish value for a given process maybe you even

(15:44) might want to hire someone on platform like upwork and ensure that the process actually works and then once you determine how the process should be executed and that it actually provides value you should automate it with eii agents which brings us to key takeaway number 15 don't think about use cases think about Roi and the formula that we use to calculate Roi is as follows on the top we have rate times amount of hours minus the operational costs divided by the development costs so the rate times amount of hours means the

(16:24) rate of an employee currently performing this process times the total number of hours that all such employees take for performing that process so it doesn't mean that it's just one employee it can be multiple employees because the number of employees definitely does affect the value of a given process and the operational costs are typically just the model costs and the server costs which in our experience again are pretty much negligible the development cost is how much it takes you to develop Your solution so let's say that for example an employee performs a process for $50 per hour and they spent on this process

(17:04) around 10 hours per week and the development cost for this process for us are at $5,000 so let me calculate this right now so a year later the ROI for this process will be at around 5.6 which means that the client has made five times the return on investment one year after we builted so again only focus on what provides the most value for the business which brings us to key takeaway number 16 agent development is an iterative process in data science competitions like kaggle where people design and train custom models for various purposes it's typically the team

(17:49) that has tested the biggest number of parameters and different model architectures that wins and the same applies to AI agents I often get get questions like how do you know how many tools to add per agent how do you know how many agents you'll need and the answer is most of the time you simply need to try it you need to try as many different architectures as possible and then compare them side by side to see which one wins only after you've had some experience building agents you'll start to see which architecture makes

(18:23) sense for a given solution so next time you are confused or your agents are underperforming make sure to build a couple of variations and test which one works better which brings us to key takeaway number 17 use divide and conquer approach divide and conquer is essentially breaking down a complex problem into manageable tasks we use this approach for almost every single agent you need to be able to deliver Solutions incrementally so instead of building an entire solution and then realizing that this is not what the client wanted make sure to split it into manageable components and deliver each

(19:02) component one by one so find an agent in a whole system that can work by itself build that agent first and deliver only that agent and only after again the client has confirmed that this agent Works proceed to building the entire system the same should apply to what you automate in a given business like for example we'd like to automate by departments first so we like to focus on a given Department Department first we like to try to automate as many solutions in a given Department as possible before transitioning to the next Department this allows us to then

(19:39) possibly combine some agents in that department together which will make our system significantly more powerful key takeaway number 18 evals are a big deal but only for big companies evals are essentially evaluation metrics that you set up for your agents to track their kpis and per performance over time evals can be extremely effective to completely eliminate your competition because they will allow you to continuously improve your Solutions over time if you have set up correct evals it means that anytime

(20:13) you deliver a similar solution to a new client all your previous Solutions get better as well additionally as the EI models are heading towards self-improvement evals will later allow your agents to possibly even self improve Over time however at the same time we found that smbs or small and medium-sized businesses might not necessarily need the evolv because small and medium-sized businesses don't have this consistent traffic of requests it doesn't really matter for them whether the solution is 5% worse or not because

(20:52) typically evals only provide incremental results you can get to 80% performance without Evol and then the last 20% is where the evolves come in so again for smbs it doesn't matter as much because the traffic of requests is much smaller they might only use the marketing agent one or a few times per day so they can easily get away without evales at the beginning but if you are working with a client who is a very large Enterprise client then definitely recommend setting up Evol from the very start key takeaway number 19 there are two types of Agents

(21:30) you can make the first one is agents and the second one is workflows yes there are actually workflows there are agentic workflows I previously haven't really considered this type of an agent I only focused on building agents for my framework and currently we don't even support workflows at all but what we found in our ageny is that sometimes there are processes that defined by gods there are some processes where the steps are predetermined for you and where the exact sequence of steps needs to be followed every single time however the steps themselves can be

(22:09) agentic so for example if you're performing a lead research process there are sometimes use cases where you need to send like three prompts to Google and the prompts themselves are the same except maybe for a company name and the agent must perform the search on the exact same websites for these companies so in this case you can actually combine standard workflows and automations with AI agents you can have a workflow where each step is agentic where it's not the whole system that's agentic but the specific steps within that process that require agentic capabilities crei for

(22:48) example is a workflow platform where you send tasks to an agent you simply hardcode them and then you send them one by one while my framework is fully agentic meaning that there's no way for you to even send tasks we did encounter a couple clients it's a very minor percentage but there are sometimes these clients that want to perform the process in the exact same way and again the steps are identic so for this we will be releasing a workflows feature very soon which brings us to key takeaway number 20 agents need to be adaptable on feedback the whole point of building

(23:24) agents is that they need to be able to interact with their environment and if your agents interact with the environment but they can't get a consistent feedback from that environment then guess what they're going to be confused so when building agents make sure that you don't just add tools for these agents that allow them to modify their environment also make sure to add tools that allow them to analyze their own results and how the impact of the previous actions actually affected their environment so for example don't just build an agent that

(23:56) can only update a database even even if your requirements do not specify this make sure to still add a tool for this agent that can then read the database records to ensure that the task has been completed successfully which brings us to key takeaway number 21 don't build around limitations this is one of the biggest mistakes you can make you need to build your agents with a stance that these models are going to be getting better and better like for example in 2023 in our agency we built a pretty complex system to avoid the context token limits but then openi released the

(24:33) 128k context model and 2 months later this system became obsolete the same might happen in the future where like for example right now openi is releasing agents like operator agent deep resarch agent and previously they also released rack which is file search and by releasing these agentic capabilities they actually made a lot of startups obsolete like for example before there were a ton of these chat with PDF apps and even though they still made significant incomes in the beginning now there are almost none of them so make sure that you don't build these obvious

(25:10) General use cases like I believe the next one might be software development so don't spend you know a year building a software developer agent if it's such an obvious use case that open thei might develop it themselves key takeaway number 22 deploying agents is a lot harder than building them so in our agent agency we started from building agents using my framework which we still use to this day of course but what we figured out shortly later on is that it would take us 2 to 3 days to build an agent but then it would take us another 3 days to deploy it which essentially means to integrate it into our client

(25:48) processes and this is why we actually decided to build our own platform so if you want to sign up for a weit list make sure to use the link below it's the only platform that we would personally use for deploying agents because no other platforms give us enough flexibility to actually do so in Productions many of them were built before even agents were a thing and many of them were simply shooting in the dark simply jumping on this hype train and not even building something that they would use themselves

(26:19) key takeaway number 23 waterfall projects don't work in our agency we only work on a subscription basis because agentic projects are too agile you cannot scope agentic projects for 3 months because they constantly evolve whenever we start working with our clients we frequently find more opportunities that we have not foreseen before so you can start from one time fees it's a always a pretty good way to get started but later on make sure you transition to a more agile service where you don't act just as an Outsource de team for your client but rather you work

(26:58) as the their partner because the goal with agent development is always not to just build an agent it's to automate a business it's to provide value to your client so make sure you have an agreement with your client where you can actually have enough room to do so key takeaway number 24 include a human in a loop for Mission critical agents sometimes there are some agents where the margin for Arrow is just so low that if the agent makes one action by mistake you will not be able to reverse back the results so for these agents you need to First include a human in a loop like for

(27:37) example you don't want to spend $110,000 on a less than ideal marketing campaign which is why in the previous video we included this human in a loop step where the client can actually review Facebook marketing campaigns in notion first later on when the agent is fully fine-tuned and when the client consistently approves every single marketing campaign you can simply remove the step key takeaway number 25 2025 is the year of vertical AI agents vertical AI agents are essentially agents that are specialized for a specific use case

(28:15) just like in B2B SAS the vertical agents are serving only a very specific type of a business customer so the same principles apply with B2B SAS it's much easier to scale you can charge significantly higher pricing because you know your customer in and out you can really fine-tune this agent to solve a very specific and a very valuable problem for a specific business and this is what makes them so easy at the start however do not start from vertical agents right away it's completely fine to build a few horizontal agents for the

(28:53) same industry that you plan to then scale in and after you've built few of these agents you will start to notice similarities between them these similarities is what you can then turn into a vertical AI agent that can be productized and adopted for many different businesses in this industry key takeaway number 26 agents don't replace people they help businesses to scale for the last key takeaway I just wanted to say that we have never seen a business owner who would immediately start firing people after their business became more efficient by automating

(29:32) businesses we simply help business owners to think bigger they think bigger they scale faster they achieve higher revenues they have more profits and most importantly their employees can now focus on higher level things and tasks that they truly enjoy so don't worry about agents replacing people I believe that Humanity will always find better things to do which will ultimately lead us into the new age of abundance and prosperity thank you for watching I will leave all of the videos I mentioned in the description and don't forget to

(30:07) subscribe