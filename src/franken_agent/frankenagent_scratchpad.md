# Implementation Notes

## Key Points
- **Path Management**
  - All paths are relative to new structure
  - Uses Path objects for cross-platform compatibility

- **Technique Management** 
  - Maintains collection of reasoning techniques
  - Techniques can be loaded, applied, modified

- **Tool Integration**
  - Each technique registered as a tool
  - Allows switching between reasoning approaches

- **System Prompt Enhancement**
  - Enhanced with available techniques info
  - Tracks currently active technique

## Improvement Suggestions
1. Technique Evaluation
   - Add mechanism to evaluate technique effectiveness
   - Enable automatic switching based on task
   - Implement technique versioning
   - Allow agent to propose/test technique modifications
   - Add validation system for new techniques

## Open Questions
1. Search Functionality
   - Should old agent search be implemented?
   - How to adapt to new structure?

2. Parameter Management  
   - How to handle temperature parameter?
   - Currently managed by base agent config

3. Query Processing
   - Implement async query from old agent?
   - Is base agent processing sufficient?

## Implementation Notes

### Tool Description Management
- Consider caching tool descriptions to avoid repeated imports
- Add validation for tool docstring format
- Implement fallback descriptions from JSON configs
- Add support for tool categories/grouping

### System Prompt Enhancement
- Consider making prompt sections configurable
- Add template support for different prompt styles
- Implement prompt versioning
- Add prompt validation/testing

### Questions to Address
1. Tool Documentation
   - How to enforce consistent docstring format?
   - Should we support markdown in docstrings?
   - How to handle tool dependencies?

2. Prompt Management
   - Should prompts be stored in separate files?
   - How to handle prompt versioning?
   - How to validate prompt effectiveness?

### Internal Tools Management
- Consider creating a decorator for internal tools
- Add validation for internal tool parameters
- Consider adding versioning for internal tools
- Add mechanism to disable/enable internal tools
- Consider adding permissions system for internal tools

### Questions to Address
3. Internal Tool Management
   - How to handle conflicts between internal and external tools?
   - Should internal tools have priority?
   - How to document internal tools consistently?
   - Should internal tools be configurable?