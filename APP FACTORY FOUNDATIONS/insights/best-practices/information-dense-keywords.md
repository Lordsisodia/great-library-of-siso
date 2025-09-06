# Information Dense Keywords for AI Prompting

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## What Are Information Dense Keywords?

**Definition**: Words that have very distinct and clear meaning for AI models
- **Examples**: create, update, delete, add, remove
- **Purpose**: Make it much clearer for AI exactly what you want and how to do it

## Bad vs Good Prompting Examples

### Bad Prompt (Vague):
```
Make the order total work better. It should handle discounts and add tax.
```

### Good Prompt (Information Dense):
```
UPDATE [filename]
- Add discount calculation functionality
- Add tax calculation functionality  
- Ensure proper error handling
ADD TEST [test_file_path]
```

## Core Information Dense Keywords

### Action Keywords:
- **CREATE** - Build something new from scratch
- **UPDATE** - Modify existing functionality  
- **DELETE** - Remove code/functionality
- **ADD** - Insert new code/features
- **REMOVE** - Take away specific elements
- **IMPLEMENT** - Build according to specifications
- **REFACTOR** - Restructure without changing behavior

### Specification Keywords:
- **ENSURE** - Make certain something works
- **VALIDATE** - Check correctness
- **TEST** - Create or run tests
- **DOCUMENT** - Add documentation
- **OPTIMIZE** - Improve performance

## Best Practices

### 1. Start with Action Keyword
```
IMPLEMENT user authentication
UPDATE database schema  
CREATE API endpoint
```

### 2. Be Specific About Files
```
UPDATE src/components/OrderTotal.tsx
ADD TEST tests/order-total.test.ts
```

### 3. Include Clear Requirements
```
CREATE payment processing service
- Handle Stripe integration
- Add error handling
- Include webhook validation
ADD INTEGRATION TEST with real Stripe test keys
```

### 4. Use for Task Breakdown
```
IMPLEMENT user dashboard
1. CREATE dashboard component
2. UPDATE routing configuration  
3. ADD user data fetching
4. CREATE responsive layout
5. ADD TEST coverage
```

## Why This Works

**Clarity**: AI understands exactly what action to take
**Consistency**: Same keywords produce similar results
**Efficiency**: Reduces back-and-forth clarification
**Precision**: Minimizes misinterpretation

## Advanced Usage

### Combining Keywords:
```
UPDATE AND TEST user login flow
- UPDATE authentication logic
- ADD error handling for edge cases
- CREATE comprehensive test suite
- VALIDATE with real user scenarios
```

### Conditional Actions:
```
IF authentication fails:
  UPDATE error messages
  ADD retry mechanism
ELSE:
  PROCEED with user dashboard
```

## Integration with 5-Step Workflow

**Step 1 (Architecture)**: CREATE architecture documents
**Step 2 (Types)**: DEFINE all type interfaces  
**Step 3 (Tests)**: ADD comprehensive test suite
**Step 4 (Implementation)**: IMPLEMENT feature logic
**Step 5 (Documentation)**: DOCUMENT architectural decisions

## Common Mistakes to Avoid

❌ **Don't**: "Make it work better"
✅ **Do**: "UPDATE error handling logic"

❌ **Don't**: "Fix the database stuff"  
✅ **Do**: "OPTIMIZE database queries for user table"

❌ **Don't**: "Add some tests"
✅ **Do**: "CREATE integration tests for payment flow"