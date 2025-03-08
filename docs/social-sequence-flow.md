sequenceDiagram
    participant Tom
    participant TomInterface as Tom's Interface
    participant JerryInterface as Jerry's Interface
    participant Jerry

    Note over Tom,Jerry: Initial Prompt Exchange
    Tom->>TomInterface: Prepare initial prompt
    TomInterface->>JerryInterface: InitialPrompt {Prompt A v1}
    JerryInterface->>Jerry: Process initial prompt
    Jerry->>JerryInterface: Evaluate prompt & prepare response
    JerryInterface->>TomInterface: EvalResponse {Eval A1, Prompt B v1}
    
    Note over Tom,Jerry: First Update Cycle
    TomInterface->>Tom: Process evaluation & prompt
    Tom->>TomInterface: Update prompt & evaluate
    TomInterface->>JerryInterface: PromptUpdate {Eval B1, Prompt A v2}
    JerryInterface->>Jerry: Process updated prompt
    Jerry->>JerryInterface: Update prompt & evaluate
    JerryInterface->>TomInterface: UpdateResponse {Eval A2, Prompt B v2}
    
    Note over Tom,Jerry: Final Evaluation
    TomInterface->>Tom: Process final update
    Tom->>TomInterface: Prepare final evaluation
    TomInterface->>JerryInterface: FinalEval {Eval B2}
    JerryInterface->>Jerry: Process final evaluation
    
    Note over Tom,Jerry: Interaction Complete