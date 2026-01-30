## Project: Building an LLM from Scratch and Applying it in Science or Engineering

You are to work on a major project to build (1) a large language model (LLM) from scratch (meaning, coding yourself
every internal mechanism of it) and (2) apply the built model in a scientific setting (see below).
You will implement a generative language model from scratch and apply it to a domain-specific task.

### Part 1: Model Implementation

Build a generative language model from scratch. You may choose either:
- Encoder-decoder architecture
- Decoder-only architecture

**Reference**: Sebastian Raschka's book on building LLMs

**Requirements**:
- Implement core components (attention mechanism, positional encoding, feedforward layers)
- Implement training loop with appropriate loss function
- Document architecture choices and hyperparameters

### Part 2: Application Task

Select ONE task from the following:

#### 2.1 Novel Molecule Generation
- Pretrain on molecular datasets ([ZINC](https://zinc.docking.org/) or [ChEMBL](https://www.ebi.ac.uk/chembl/) database)
- Generate valid molecular structures
- Evaluate generated molecules for chemical validity

#### 2.2 SMILES to SELFIES Translation
- Translate between molecular string representations
- Validate output SELFIES strings
- Measure translation accuracy

#### 2.3 Multilingual Text Generation
- Use AI4Bharat dataset from HuggingFace: [see here](https://huggingface.co/ai4bharat)
- Generate coherent text in target language(s)
- Evaluate generation quality

#### 2.4 Language Translation
- Implement translation between language pair of your choice
- Evaluate translation quality using standard metrics (BLEU, METEOR, etc.)

### Team size
- You can work alone (which is not advisable) or in a group of a minimum of 2 or a maximum of 3 students. If you work
   in a group then each student in the group will be judged based on the their group performance in the project.

### Deliverables

1. Source code (well-documented, version-controlled)
2. Technical report documenting:
   - Architecture details
   - Training procedure and hyperparameters
   - Task-specific implementation
   - Results and evaluation metrics
   - Challenges and solutions
3. Trained model checkpoint
4. Demo/results on test cases

### Evaluation Criteria

- Model implementation correctness
- Code quality and documentation
- Task performance
- Technical report clarity and depth
- Novelty in approach (if any)
- Each of the above will be judged in a final presentation.

### Weight

Total: 20. 
But, we may introduce a BONUS component (10% = 2) which depends on many factors: TBA.
