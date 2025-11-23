# LangChain Tools Quick Reference

## Overview

LangChain provides 1000+ integrations through tools and toolkits. Tools enable agents to interact with external services, each with descriptions that guide agent selection.

---

## Core Tool Categories

### Search & Web
- **BraveSearch, DuckDuckGoSearch, SerpAPI** - Web search engines
- **WikipediaSearch, TavilySearch** - Knowledge base search
- **GoogleSearch, GoogleLens, GooglePlaces** - Google services
- **GoogleFinance, GoogleTrends, GoogleBooks** - Specialized Google APIs

### AI & ML Services
- **OpenAIDALLEImageGenerator, StableDiffusionImageGenerator** - Image generation
- **ElevenLabsText2Speech, AzureAITextToSpeech** - Text-to-speech
- **AzureAISpeechToText, EdenAISpeechToText** - Speech recognition
- **SceneXplain, AzureAIImageAnalysis** - Image analysis
- **EdenAI suite** - Content moderation, object detection, document parsing

### Data & Databases
- **SQLDatabaseQuery, SQLDatabaseMetadata** - SQL operations
- **CassandraQueryTool, SparkSQLQuery** - NoSQL and big data
- **PowerBIQuery** - Business intelligence
- **VectorStoreQATool** - Vector database queries

### API & Web Interaction
- **RequestsGetTool, RequestsPostTool** - HTTP requests
- **BaseGraphQLTool, OpenAPITool** - API integrations
- **Playwright suite** - Browser automation (navigate, click, extract text/links)

### File Operations
- **ReadFileTool, WriteFileTool** - File I/O
- **CopyFileTool, MoveFileTool, DeleteFileTool** - File management
- **ListDirectoryTool, FileSearchTool** - Directory operations

### Communication
- **Gmail**: Send messages, search, create drafts
- **Slack**: Send/schedule messages, retrieve channel info
- **Office365**: Email and calendar management

### Developer Tools
- **GitHubTool, GitLabTool** - Repository management
- **JiraTool, ClickUpTool** - Project management
- **ShellTool, PythonREPLTool** - Code execution
- **E2BDataAnalysisTool** - Data analysis sandbox

### Finance & Business
- **YahooFinanceNewsTool** - Financial news
- **Polygon suite** - Stock quotes, financials, aggregates
- **Company financials** - Balance sheets, cash flow, income statements

### Knowledge & Research
- **ArxivTool, PubMedTool, SemanticScholarTool** - Academic papers
- **WikidataTool, WolframAlphaTool** - Structured knowledge
- **MerriamWebsterTool, StackExchangeTool** - Reference lookups

### Specialized
- **OpenWeatherMapTool** - Weather data
- **MultionSession suite** - Browser session management
- **ZapierNLATools, IFTTTWebhook** - Automation integrations
- **HumanInputRun** - Request user input during execution

---

## Toolkits (Pre-configured Collections)

Toolkits bundle related tools for specific domains:

- **Development**: JiraToolkit, GitHubToolkit, GitLabToolkit
- **Communication**: GmailToolkit, SlackToolkit, Office365Toolkit
- **Data**: SQLDatabaseToolkit, SparkSQLToolkit, PowerBIToolkit, CassandraToolkit
- **Web**: PlaywrightBrowserToolkit, OpenAPIToolkit
- **File System**: FileManagementToolkit
- **AI Services**: AzureAIToolkit, EdenAIToolkit
- **Finance**: PolygonToolkit, YahooFinanceToolkit

---

## Tool Structure

```
BaseTool
├── Tool (custom tools)
└── Specialized Tools (BraveSearch, etc.)
```

**Key Features:**
- Name and description for agent selection
- Synchronous and asynchronous execution support
- Callback management for monitoring

---

## Implementation Notes

- Most tools require API keys or credentials
- Available via `langchain-community` package
- Tools can be used individually or bundled in toolkits
- Agents use tool descriptions to select appropriate tools

**Resources:**
- Docs: https://python.langchain.com/docs/integrations/tools/
- API Reference: https://python.langchain.com/api_reference/community/tools.html