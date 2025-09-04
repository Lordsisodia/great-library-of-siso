# 🛒 XP Spending Economy - Earned Indulgence System

## 🧠 Behavioral Psychology Foundation

### Core Principles
- **Token Economy Psychology**: Earned rewards feel more valuable than given rewards
- **Loss Aversion**: Having to "spend" XP makes you consider if reward is worth it
- **Delayed Gratification**: Building XP to "afford" bigger rewards 
- **Self-Regulation**: Gamifying the balance between work and indulgence
- **Hyperbolic Discounting**: Immediate costs (XP spending) vs delayed pleasure

### Research-Backed Benefits
- Creates self-control through earned pleasure mechanism
- Transforms impulses into conscious decisions
- Makes vices feel "earned" rather than guilty
- Builds sustainable work-life balance
- Reduces addiction potential through XP budgeting

## 🎯 XP Spending Categories & Pricing

### 💨 Cannabis & Relaxation (High Value Indulgences)
```typescript
enum CannabiRewards {
  MICRO_DOSE = 150,     // "Study high" - 30min creative session
  CHILL_SESSION = 300,  // 2-hour relaxation period
  DEEP_HIGH = 500,      // 4-hour complete relaxation day
  WEEKEND_PASS = 800,   // Full weekend smoking permission
  STRAIN_UPGRADE = 200, // Premium strain selection
  EDIBLE_TREAT = 250,   // Special edible indulgence
}
```

### 🏖️ Time Off & Breaks
```typescript
enum TimeOffRewards {
  POWER_NAP = 100,      // 20min guilt-free nap
  LONG_BREAK = 200,     // 2-hour extended break
  HALF_DAY = 600,       // Half day off work
  FULL_DAY = 1200,      // Complete day off
  WEEKEND_LAZY = 800,   // Guilt-free lazy weekend
  NETFLIX_BINGE = 400,  // 6-hour binge session
  SOCIAL_MEDIA = 150,   // 1-hour scrolling permission
}
```

### 🍟 Food & Treats  
```typescript
enum FoodRewards {
  JUNK_FOOD = 100,      // Guilt-free fast food meal
  EXPENSIVE_MEAL = 300, // High-end restaurant visit
  DESSERT_BINGE = 200,  // Ice cream/sweets session
  ALCOHOL = 250,        // Drinking session permission
  CHEAT_DAY = 500,      // Full day dietary freedom
  UBER_EATS = 150,      // Lazy meal delivery
}
```

### 🎮 Entertainment & Hobbies
```typescript
enum EntertainmentRewards {
  GAMING_SESSION = 300, // 4-hour gaming marathon
  MOVIE_THEATER = 200,  // Cinema experience
  CONCERT_TICKET = 800, // Live music event
  HOBBY_SPLURGE = 400,  // Equipment/supplies purchase
  STREAMING_DAY = 300,  // All-day content consumption
  SOCIAL_EVENT = 350,   // Party/club night out
}
```

### 💰 Purchases & Luxuries
```typescript
enum PurchaseRewards {
  SMALL_LUXURY = 400,   // $50-100 non-essential purchase
  MEDIUM_LUXURY = 800,  // $200-300 wanted item
  BIG_LUXURY = 1500,    // $500+ significant purchase
  SUBSCRIPTION = 200,   // New streaming/app subscription
  CLOTHING = 300,       // Non-essential clothing item
  GADGET_UPGRADE = 1000, // Tech upgrade/new device
}
```

## 🏪 Advanced XP Store Features

### 1. Dynamic Pricing System
```typescript
interface DynamicPricing {
  basePrice: number
  demandMultiplier: number    // Higher when used frequently
  streakDiscount: number      // Discount for long productivity streaks
  seasonalPricing: number     // Holiday/weekend adjustments
  bundleDeals: BundleOffer[]  // Combo purchases
}

// Example: Cannabis pricing increases if used too often
const calculatePrice = (reward: Reward, userHistory: UsageHistory) => {
  let price = reward.basePrice
  
  // Demand multiplier (addiction prevention)
  if (userHistory.lastWeekUsage > 3) price *= 1.5
  
  // Streak discount (reward consistency)
  if (userHistory.currentStreak > 10) price *= 0.8
  
  return Math.round(price)
}
```

### 2. XP Loan System (Dangerous but Effective)
```typescript
interface XPLoan {
  amount: number
  interestRate: number        // 20% interest on borrowed XP
  dueDate: Date
  collateral: string          // What you lose if not repaid
}

// Example: Borrow XP for immediate reward, pay back with interest
const borrowXP = (amount: number) => ({
  amount,
  interestRate: 0.2,          // Must earn 20% more XP to repay
  dueDate: addDays(new Date(), 7),
  penalty: 'lose_streak'      // Harsh penalty for default
})
```

### 3. Subscription Rewards
```typescript
enum SubscriptionRewards {
  DAILY_SMOKE = 2000,    // 30 days of daily smoke permission
  WEEKEND_FREEDOM = 1500, // 4 weekends of complete freedom
  MONTHLY_LUXURY = 3000,  // $200/month luxury budget
  GAMING_PASS = 1800,     // Unlimited gaming for 30 days
}
```

### 4. Bundle Deals & Combos
```typescript
const bundleDeals = [
  {
    name: "Weekend Warrior",
    items: [CannabiRewards.WEEKEND_PASS, TimeOffRewards.WEEKEND_LAZY],
    normalPrice: 1600,
    bundlePrice: 1200,        // 25% discount
    description: "Perfect lazy stoned weekend"
  },
  {
    name: "Productivity Recovery",
    items: [TimeOffRewards.HALF_DAY, FoodRewards.EXPENSIVE_MEAL, CannabiRewards.CHILL_SESSION],
    normalPrice: 1100,
    bundlePrice: 900,
    description: "Recover from intense work period"
  }
]
```

## 🧮 Economic Balance Design

### XP Earning vs Spending Ratios
```typescript
const economicBalance = {
  dailyEarningPotential: 500,     // Max XP per day
  averageDailySpending: 300,      // Sustainable spending
  weeklyDeficit: 1400,            // Forces productive weekdays
  luxuryEarningTime: "3-5 days",  // Time to afford big rewards
  emergencyReserve: 1000          // Minimum XP buffer recommended
}
```

### Pricing Psychology
- **Cannabis rewards**: Expensive to prevent daily use, encourage moderation
- **Time off**: Moderately priced, accessible for daily breaks
- **Food treats**: Cheap enough for regular small indulgences  
- **Luxury purchases**: Very expensive, require significant productivity
- **Loans**: Available but punitive, emergency use only

## 💡 Advanced Psychological Features

### 1. Spending Guilt Elimination
```typescript
const showSpendingJustification = (purchase: Reward, xpEarned: number) => {
  return `You EARNED this ${purchase.name} through:
  - ${xpEarned}XP worth of productive work
  - ${calculateHours(xpEarned)} hours of focused effort
  - You deserve this reward!`
}
```

### 2. Spending Anxiety (Positive Stress)
```typescript
const showSpendingCost = (reward: Reward, currentXP: number) => {
  const remaining = currentXP - reward.price
  
  return {
    message: `This will cost ${reward.price}XP`,
    warning: remaining < 200 ? "This will drain your XP reserves!" : null,
    timeToRecover: `${Math.ceil((reward.price / 150))} days to re-earn this amount`,
    opportunity: "Consider smaller reward or earn more XP first?"
  }
}
```

### 3. Social Pressure Simulation
```typescript
const socialPressureFeatures = {
  pastSelfComparison: "Last month you earned this reward in 3 days...",
  productivity: "High performers earn luxury rewards 2x faster",
  efficiency: "You could afford this tomorrow with just 4 hours of deep work",
  streak: "Your 15-day streak deserves a proper celebration!"
}
```

## 📊 Implementation Schema

### Database Models
```typescript
// Add to existing schema
model XPPurchase {
  id          String   @id @default(cuid())
  userId      String
  rewardType  RewardCategory
  rewardName  String
  xpCost      Int
  actualPrice Int      // After discounts/multipliers
  purchasedAt DateTime @default(now())
  consumed    Boolean  @default(false)
  consumedAt  DateTime?
  
  // Loan system
  isLoan      Boolean  @default(false)
  loanAmount  Int?
  loanDue     DateTime?
  loanRepaid  Boolean  @default(false)
  
  user        User     @relation(fields: [userId], references: [id])
}

model XPBalance {
  id            String @id @default(cuid())
  userId        String @unique
  currentXP     Int    @default(0)
  totalEarned   Int    @default(0)
  totalSpent    Int    @default(0)
  pendingLoans  Int    @default(0)
  
  user          User   @relation(fields: [userId], references: [id])
}

enum RewardCategory {
  CANNABIS
  TIME_OFF
  FOOD
  ENTERTAINMENT  
  LUXURY_PURCHASE
  SUBSCRIPTION
}
```

This system transforms your XP into a **personal vice currency** where indulgences must be earned through productivity, creating sustainable work-life balance through gamified self-control! 🎮💰