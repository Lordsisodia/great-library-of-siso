# 📋 Component Specifications

Exact specifications for building consistent UI components across all applications.

## 🔘 Buttons

### **Primary Button**
```css
/* Size & Spacing */
min-height: 44px;
padding: 12px 24px;
border-radius: 6px;

/* Typography */
font-size: 16px;
font-weight: 600;
text-align: center;

/* Colors */
background: #007bff;
color: #ffffff;
border: none;

/* States */
:hover { background: #0056b3; }
:active { background: #004494; }
:disabled { background: #6c757d; opacity: 0.6; }
```

### **Secondary Button**
```css
/* Same sizing as primary */
background: transparent;
color: #007bff;
border: 2px solid #007bff;

/* States */
:hover { background: #007bff; color: #ffffff; }
```

### **Button Guidelines**
- ✅ Use descriptive text ("Download Report" not "Click Here")
- ✅ Maximum 2-3 words when possible
- ✅ Only one primary button per section
- ❌ Don't use all caps unless brand requires it
- ❌ Don't make buttons smaller than 44px height

## 📝 Form Inputs

### **Text Input**
```css
/* Size & Spacing */
min-height: 44px;
padding: 12px 16px;
border-radius: 4px;

/* Typography */
font-size: 16px;
line-height: 1.4;

/* Colors */
background: #ffffff;
border: 2px solid #e1e5e9;
color: #333333;

/* States */
:focus { border-color: #007bff; outline: none; }
:error { border-color: #dc3545; }
:disabled { background: #f8f9fa; }
```

### **Label Specifications**
```css
font-size: 14px;
font-weight: 500;
color: #495057;
margin-bottom: 4px;
display: block;
```

### **Input Guidelines**
- ✅ Always include visible labels
- ✅ Use appropriate input types (email, tel, etc.)
- ✅ Show validation errors immediately
- ❌ Don't hide labels inside inputs only
- ❌ Don't use vague error messages

## 📊 Cards

### **Standard Card**
```css
/* Structure */
background: #ffffff;
border-radius: 8px;
box-shadow: 0 2px 4px rgba(0,0,0,0.1);
padding: 24px;

/* Spacing */
margin-bottom: 16px;

/* Content Structure */
.card-header { margin-bottom: 16px; }
.card-body { margin-bottom: 16px; }
.card-footer { margin-top: 16px; }
```

### **Card Guidelines**
- ✅ Use consistent padding throughout
- ✅ Include hover states for interactive cards
- ✅ Group related information together
- ❌ Don't nest cards within cards
- ❌ Don't make cards too wide (max 400px usually)

## 🧭 Navigation

### **Primary Navigation**
```css
/* Desktop */
height: 64px;
background: #ffffff;
border-bottom: 1px solid #e1e5e9;
padding: 0 24px;

/* Mobile */
@media (max-width: 768px) {
  height: 56px;
  padding: 0 16px;
}
```

### **Navigation Links**
```css
font-size: 16px;
font-weight: 500;
color: #495057;
text-decoration: none;
padding: 8px 12px;

/* States */
:hover { color: #007bff; }
.active { color: #007bff; font-weight: 600; }
```

### **Navigation Guidelines**
- ✅ Keep navigation items to 7 or fewer
- ✅ Highlight current page/section clearly
- ✅ Use familiar icons for common actions
- ❌ Don't hide navigation without good reason
- ❌ Don't use dropdown menus on mobile

## 🚨 Alerts & Messages

### **Success Alert**
```css
background: #d4edda;
border: 1px solid #c3e6cb;
color: #155724;
padding: 12px 16px;
border-radius: 4px;
margin-bottom: 16px;
```

### **Error Alert**
```css
background: #f8d7da;
border: 1px solid #f5c6cb;
color: #721c24;
/* Same sizing as success */
```

### **Alert Guidelines**
- ✅ Use appropriate colors for message type
- ✅ Include clear actions users can take
- ✅ Auto-dismiss success messages after 5 seconds
- ❌ Don't rely only on color to convey meaning
- ❌ Don't use technical error messages

## 📋 Data Tables

### **Table Specifications**
```css
/* Table */
width: 100%;
border-collapse: collapse;
background: #ffffff;

/* Headers */
th {
  padding: 16px;
  text-align: left;
  font-weight: 600;
  background: #f8f9fa;
  border-bottom: 2px solid #dee2e6;
}

/* Cells */
td {
  padding: 16px;
  border-bottom: 1px solid #dee2e6;
}

/* Rows */
tbody tr:hover {
  background: #f8f9fa;
}
```

### **Table Guidelines**
- ✅ Left-align text, right-align numbers
- ✅ Use zebra striping for long tables
- ✅ Make tables horizontally scrollable on mobile
- ❌ Don't put too much content in table cells
- ❌ Don't use tables for layout

## 🔧 Spacing System

### **Standard Spacing Units**
```css
/* Base unit: 4px */
--space-xs: 4px;
--space-sm: 8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;
--space-xxl: 48px;
--space-xxxl: 64px;
```

### **Application Guidelines**
- ✅ Use consistent spacing units throughout
- ✅ Double spacing between unrelated sections
- ✅ Use smaller spacing within component groups
- ❌ Don't use random spacing values
- ❌ Don't cram elements together without breathing room

## 🎨 Color Specifications

### **Primary Palette**
```css
/* Blues */
--primary-50: #e3f2fd;
--primary-500: #007bff;  /* Main brand color */
--primary-700: #0056b3;
--primary-900: #004494;

/* Grays */
--gray-50: #f8f9fa;
--gray-100: #e9ecef;
--gray-300: #dee2e6;
--gray-500: #6c757d;
--gray-700: #495057;
--gray-900: #343a40;

/* Status Colors */
--success: #28a745;
--warning: #ffc107;
--error: #dc3545;
--info: #17a2b8;
```

### **Color Usage Rules**
- ✅ Use primary colors for key actions
- ✅ Use gray shades for supporting content
- ✅ Maintain 4.5:1 contrast ratio for text
- ❌ Don't use more than 5 colors in one interface
- ❌ Don't use pure black (#000000) for text

## 🔍 Interactive States

### **Required States for All Interactive Elements**
```css
/* Default state - how it normally looks */
.element { }

/* Hover - mouse over (desktop only) */
.element:hover { }

/* Focus - keyboard navigation */
.element:focus { 
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

/* Active - being clicked/tapped */
.element:active { }

/* Disabled - not available */
.element:disabled { 
  opacity: 0.6;
  cursor: not-allowed;
}
```

### **State Guidelines**
- ✅ Every interactive element needs all states
- ✅ Make focus states clearly visible
- ✅ Use consistent hover effects throughout
- ❌ Don't use hover effects on mobile
- ❌ Don't forget disabled states

---

## 🚀 Implementation Checklist

**Before building any component:**
- [ ] Does it follow the established spacing system?
- [ ] Are all interactive states defined?
- [ ] Does it meet accessibility contrast requirements?
- [ ] Is it responsive across all breakpoints?
- [ ] Have you tested it with keyboard navigation?
- [ ] Does it match the established visual hierarchy?

**Remember: These specifications ensure consistency and quality across all applications. Follow them unless you have a compelling reason to deviate.**