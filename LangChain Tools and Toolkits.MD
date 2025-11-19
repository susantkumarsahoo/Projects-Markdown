# LangChain Tools and Toolkits Reference

## Overview

LangChain offers an extensive ecosystem with over 1000 integrations across various components. Tools are classes that agents use to interact with the world, each with a description that helps agents choose the right tool for the job.

---

## Individual Tools

### Search & Web Tools

- **BraveSearch** - Search using Brave Search API
- **DuckDuckGoSearch** - DuckDuckGo search integration
- **SerpAPI** - Google search via SerpAPI
- **GoogleSearch** - Google search API integration
- **WikipediaSearch** - Wikipedia API search
- **TavilySearch** - Tavily search service
- **SearchApiTool** - SearchApi.io integration
- **SearxSearch** - Searx instance search
- **JinaSearch** - Jina search API
- **YouSearch** - You.com API search
- **GoogleLens** - Google Lens API
- **GooglePlaces** - Google Places search
- **GoogleJobs** - Google Jobs API
- **GoogleFinance** - Google Finance data
- **GoogleTrends** - Google Trends API
- **GoogleBooks** - Google Books API search
- **AskNews** - AskNews API search

### AI & ML Services

- **OpenAIDALLEImageGenerator** - Generate images using DALL-E
- **StableDiffusionImageGenerator** - Stable Diffusion image generation
- **ElevenLabsText2Speech** - Text-to-speech conversion
- **AzureAISpeechToText** - Azure speech recognition
- **AzureAITextToSpeech** - Azure text-to-speech
- **EdenAISpeechToText** - Eden AI speech recognition
- **EdenAITextToSpeech** - Eden AI text-to-speech
- **EdenAIExplicitImageDetection** - Image content moderation
- **EdenAIObjectDetection** - Object detection in images
- **EdenAIIdentityParsing** - Identity document parsing
- **EdenAIInvoiceParsing** - Invoice data extraction
- **EdenAIExplicitTextDetection** - Text content moderation
- **HuggingFaceTextToSpeech** - HuggingFace TTS models
- **SceneXplain** - Image explanation and analysis
- **AzureAIDocumentIntelligence** - Document analysis
- **AzureAIImageAnalysis** - Image analysis services

### Data & Database Tools

- **SQLDatabaseQuery** - Query SQL databases
- **SQLDatabaseMetadata** - Get database metadata
- **SQLDatabaseListTables** - List database tables
- **SQLDatabaseCheckQuery** - Validate SQL queries with LLM
- **CassandraQueryTool** - Query Cassandra databases
- **CassandraGetKeyspaceSchema** - Get Cassandra schema
- **CassandraGetTableData** - Retrieve table data
- **SparkSQLQuery** - Query Spark SQL
- **SparkSQLMetadata** - Spark SQL metadata
- **SparkSQLListTables** - List Spark tables
- **PowerBIQuery** - Query Power BI datasets
- **PowerBIMetadata** - Power BI dataset metadata
- **PowerBIListTables** - List Power BI tables
- **VectorStoreQATool** - Query vector databases
- **VectorStoreQAWithSourcesTool** - Query with source tracking
- **DatahheraldTool** - Dataherald SDK integration

### API & Web Interaction

- **RequestsGetTool** - Make GET requests
- **RequestsPostTool** - Make POST requests
- **RequestsPutTool** - Make PUT requests
- **RequestsPatchTool** - Make PATCH requests
- **RequestsDeleteTool** - Make DELETE requests
- **BaseGraphQLTool** - GraphQL API queries
- **OpenAPITool** - OpenAPI specification tools
- **AIPluginTool** - AI plugin integration
- **PlaywrightNavigateTool** - Navigate to URLs
- **PlaywrightClickTool** - Click page elements
- **PlaywrightExtractTextTool** - Extract page text
- **PlaywrightExtractHyperlinksTool** - Extract links
- **PlaywrightGetElementsTool** - Get page elements
- **PlaywrightCurrentWebPageTool** - Get current page
- **PlaywrightNavigateBackTool** - Navigate back

### File Operations

- **ReadFileTool** - Read file contents
- **WriteFileTool** - Write to files
- **CopyFileTool** - Copy files
- **DeleteFileTool** - Delete files
- **MoveFileTool** - Move files
- **ListDirectoryTool** - List directory contents
- **FileSearchTool** - Search for files with regex

### Communication Tools

#### Gmail Tools
- **GmailSendMessage** - Send emails
- **GmailSearch** - Search messages/threads
- **GmailGetMessage** - Get message by ID
- **GmailGetThread** - Get thread by ID
- **GmailCreateDraft** - Create draft emails

#### Slack Tools
- **SlackSendMessage** - Send Slack messages
- **SlackScheduleMessage** - Schedule messages
- **SlackGetMessages** - Retrieve messages
- **SlackGetChannel** - Get channel information

#### Office365 Tools
- **Office365SendEmail** - Send emails
- **Office365SearchEmails** - Search emails
- **Office365CreateDraft** - Create draft emails
- **Office365SendEvent** - Create calendar events
- **Office365SearchEvents** - Search calendar events

### Developer & Productivity Tools

- **GitHubTool** - GitHub API interactions
- **GitLabTool** - GitLab API interactions
- **JiraTool** - Atlassian Jira operations
- **ClickUpTool** - ClickUp task management
- **ShellTool** - Execute shell commands
- **PythonREPLTool** - Execute Python code
- **E2BDataAnalysisTool** - Data analysis sandbox
- **BearlyInterpreterTool** - Code evaluation sandbox
- **RizaCodeInterpreter** - JavaScript execution

### Finance & Business

- **YahooFinanceNewsTool** - Financial news search
- **PolygonAggregates** - Stock price aggregates
- **PolygonFinancials** - Company financials
- **PolygonLastQuote** - Latest stock quotes
- **PolygonTickerNews** - Ticker news
- **BalanceSheets** - Company balance sheets
- **CashFlowStatements** - Cash flow data
- **IncomeStatements** - Income statement data
- **DataForSeoGoogleSearch** - SEO data from Google

### Knowledge & Research

- **ArxivTool** - Search Arxiv papers
- **PubMedTool** - Search PubMed database
- **SemanticScholarTool** - Academic paper search
- **WikidataTool** - Wikidata queries
- **WolframAlphaTool** - Wolfram Alpha queries
- **MerriamWebsterTool** - Dictionary lookups
- **StackExchangeTool** - Stack Exchange search
- **GoldenQueryTool** - Golden API queries

### Specialized Tools

- **OpenWeatherMapTool** - Weather information
- **PassioNutritionAI** - Nutrition data lookup
- **SteamSearch** - Steam game search
- **RedditSearchTool** - Reddit post search
- **YouTubeTool** - YouTube search
- **MultionCreateSession** - Create browser sessions
- **MultionUpdateSession** - Update browser sessions
- **MultionCloseSession** - Close browser sessions
- **NucliaUnderstanding** - File processing with Nuclia
- **CogniswitchKnowledge** - Cogniswitch question answering
- **CogniswitchStoreData** - Store data via Cogniswitch
- **ConneryActionTool** - Connery action execution
- **IFTTTWebhook** - IFTTT webhook integration
- **ZapierNLATools** - Zapier action execution
- **HumanInputRun** - Request user input
- **SleepTool** - Add delays

---

## Toolkits (Collections of Related Tools)

Toolkits are curated collections of tools designed to work together for specific domains:

### Development & Project Management
- **JiraToolkit** - Complete Jira operations suite
- **GitHubToolkit** - GitHub repository management
- **GitLabToolkit** - GitLab operations
- **ClickUpToolkit** - ClickUp project management

### Communication
- **GmailToolkit** - Complete Gmail operations
- **SlackToolkit** - Slack messaging and management
- **Office365Toolkit** - Microsoft Office integrations

### Data & Analytics
- **SQLDatabaseToolkit** - Comprehensive SQL operations
- **SparkSQLToolkit** - Spark data processing
- **PowerBIToolkit** - Power BI dataset interactions
- **CassandraToolkit** - Cassandra database operations
- **DatabricksUCFunctionToolkit** - Databricks Unity Catalog

### Web & Browser
- **PlaywrightBrowserToolkit** - Browser automation suite
- **OpenAPIToolkit** - OpenAPI/REST API interactions

### File & System
- **FileManagementToolkit** - File system operations

### AI Services
- **AzureAIToolkit** - Azure AI services
- **EdenAIToolkit** - Eden AI service suite
- **AmadeusToolkit** - Amadeus travel services

### Finance
- **PolygonToolkit** - Polygon.io financial data
- **YahooFinanceToolkit** - Yahoo Finance tools

---

## Tool Class Hierarchy

```
ToolMetaclass
└── BaseTool
    ├── Tool (custom tools)
    └── Specialized Tools (BraveSearch, HumanInputRun, etc.)
```

## Key Helpers

- **CallbackManagerForToolRun** - Synchronous callback management
- **AsyncCallbackManagerForToolRun** - Asynchronous callback management

---

## Usage Notes

- Each tool has a name and description that agents use for selection
- Tools can be used individually or as part of toolkits
- Community integrations are available via the `langchain-community` package
- Many tools require API keys or authentication credentials
- Tools support both synchronous and asynchronous execution

---

## Resources

- Official Documentation: https://python.langchain.com/docs/integrations/tools/
- API Reference: https://python.langchain.com/api_reference/community/tools.html
- Community Package: `langchain-community`

---

*Last Updated:
*This reference includes tools from LangChain Community package and official integrations.*