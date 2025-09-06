#!/usr/bin/env python3
"""
Enterprise Claude Configuration System Validator
Validates all @include references, YAML anchors, and system integrity
"""

import os
import re
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class EnterpriseSystemValidator:
    def __init__(self, claude_dir: str = "/Users/shaansisodia/.claude"):
        self.claude_dir = Path(claude_dir)
        self.main_config = self.claude_dir / "CLAUDE.md"
        self.shared_dir = self.claude_dir / "shared"
        self.commands_shared_dir = self.claude_dir / "commands/shared"
        
        self.include_pattern = re.compile(r'@include\s+([^\s#]+)(?:#([^\s]+))?')
        self.anchor_pattern = re.compile(r'^([^:\s]+):\s*&([^\s]+)', re.MULTILINE)
        
        self.results = {
            'total_includes': 0,
            'resolved_includes': 0,
            'missing_files': [],
            'missing_anchors': [],
            'valid_files': [],
            'system_health': 'UNKNOWN'
        }
    
    def validate_system(self) -> Dict:
        """Main validation pipeline"""
        print("🚀 Enterprise Claude System Validation Starting...")
        print(f"📁 Base Directory: {self.claude_dir}")
        
        # Step 1: Validate main configuration
        if not self.main_config.exists():
            self.results['system_health'] = 'CRITICAL'
            return self.results
        
        # Step 2: Parse @include references
        includes = self.parse_includes()
        self.results['total_includes'] = len(includes)
        
        # Step 3: Validate each reference
        for file_path, anchor in includes:
            self.validate_include(file_path, anchor)
        
        # Step 4: Calculate health score
        self.calculate_health_score()
        
        # Step 5: Generate report
        self.generate_report()
        
        return self.results
    
    def parse_includes(self) -> List[Tuple[str, str]]:
        """Parse all @include references from CLAUDE.md"""
        includes = []
        
        with open(self.main_config, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = self.include_pattern.findall(content)
        for file_path, anchor in matches:
            includes.append((file_path, anchor))
        
        print(f"📋 Found {len(includes)} @include references")
        return includes
    
    def validate_include(self, file_path: str, anchor: str) -> bool:
        """Validate single @include reference"""
        # Resolve full path
        full_path = self.claude_dir / file_path
        
        # Check if file exists
        if not full_path.exists():
            self.results['missing_files'].append(file_path)
            print(f"❌ Missing file: {file_path}")
            return False
        
        # Check if anchor exists (if specified)
        if anchor:
            if not self.validate_anchor(full_path, anchor):
                self.results['missing_anchors'].append(f"{file_path}#{anchor}")
                print(f"⚠️  Missing anchor: {file_path}#{anchor}")
                return False
        
        self.results['resolved_includes'] += 1
        self.results['valid_files'].append(file_path)
        print(f"✅ Valid: {file_path}#{anchor}")
        return True
    
    def validate_anchor(self, file_path: Path, anchor_name: str) -> bool:
        """Check if YAML anchor exists in file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all anchors in file
            anchors = self.anchor_pattern.findall(content)
            anchor_names = [name for _, name in anchors]
            
            return anchor_name in anchor_names
            
        except Exception as e:
            print(f"🔥 Error reading {file_path}: {e}")
            return False
    
    def calculate_health_score(self):
        """Calculate overall system health"""
        total = self.results['total_includes']
        resolved = self.results['resolved_includes']
        
        if total == 0:
            health_score = 0
        else:
            health_score = (resolved / total) * 100
        
        if health_score >= 95:
            self.results['system_health'] = 'EXCELLENT'
        elif health_score >= 85:
            self.results['system_health'] = 'GOOD'
        elif health_score >= 70:
            self.results['system_health'] = 'WARNING'
        else:
            self.results['system_health'] = 'CRITICAL'
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("🎯 ENTERPRISE SYSTEM VALIDATION REPORT")
        print("="*60)
        
        print(f"📊 System Health: {self.results['system_health']}")
        print(f"📋 Total @include References: {self.results['total_includes']}")
        print(f"✅ Resolved Successfully: {self.results['resolved_includes']}")
        print(f"❌ Missing Files: {len(self.results['missing_files'])}")
        print(f"⚠️  Missing Anchors: {len(self.results['missing_anchors'])}")
        
        if self.results['missing_files']:
            print("\n🔥 MISSING FILES:")
            for file in self.results['missing_files']:
                print(f"   ❌ {file}")
        
        if self.results['missing_anchors']:
            print("\n⚠️  MISSING ANCHORS:")
            for anchor in self.results['missing_anchors']:
                print(f"   ⚠️  {anchor}")
        
        print(f"\n🎯 Resolution Rate: {self.results['resolved_includes']}/{self.results['total_includes']} ({(self.results['resolved_includes']/max(self.results['total_includes'],1)*100):.1f}%)")
        
        # System recommendations
        self.generate_recommendations()
    
    def generate_recommendations(self):
        """Generate system improvement recommendations"""
        print("\n📋 ENTERPRISE RECOMMENDATIONS:")
        
        if self.results['system_health'] == 'EXCELLENT':
            print("🚀 SYSTEM STATUS: ENTERPRISE-READY")
            print("   ✅ All configurations validated")
            print("   ✅ Ready for team deployment")
            print("   ✅ Consider implementing CI/CD validation")
        
        elif self.results['missing_files']:
            print("🔧 PRIORITY ACTIONS:")
            print("   1. Create missing configuration files")
            print("   2. Implement file generation scripts")
            print("   3. Add file existence validation")
        
        elif self.results['missing_anchors']:
            print("🔧 PRIORITY ACTIONS:")
            print("   1. Add missing YAML anchors")
            print("   2. Verify anchor naming consistency")
            print("   3. Implement anchor validation")
        
        print("\n🎯 NEXT STEPS:")
        print("   • Set up automated validation in CI/CD")
        print("   • Create team deployment guidelines")
        print("   • Implement configuration versioning")
        print("   • Build enterprise monitoring dashboard")

def main():
    """Run enterprise system validation"""
    try:
        validator = EnterpriseSystemValidator()
        results = validator.validate_system()
        
        # Exit with appropriate code
        if results['system_health'] in ['EXCELLENT', 'GOOD']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"🔥 CRITICAL ERROR: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()