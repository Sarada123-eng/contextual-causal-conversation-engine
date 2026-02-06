"""Normalize causal factors into unified categories."""

from __future__ import annotations

from typing import Dict, Set


class FactorNormalizer:
    """Normalize similar causal factors into unified categories."""
    
    # Canonical factor categories
    PRODUCT_DEFECTS = "Product Defects"
    BILLING_ERRORS = "Billing Errors"
    DELIVERY_DELAYS = "Delivery Delays"
    POOR_QUALITY = "Poor Product Quality"
    WRONG_ITEM = "Wrong Item Sent"
    MISSING_ITEMS = "Missing Items"
    POOR_SERVICE = "Poor Customer Service"
    PRICING_ISSUES = "Pricing Issues"
    TECHNICAL_ISSUES = "Technical Issues"
    SHIPPING_DAMAGE = "Shipping Damage"
    RETURN_ISSUES = "Return Issues"
    ACCOUNT_ISSUES = "Account Issues"
    GENERAL_ISSUE = "General Issues"
    
    def __init__(self):
        """Initialize factor normalization mappings."""
        self.normalization_map: Dict[str, str] = self._build_normalization_map()
    
    def _build_normalization_map(self) -> Dict[str, str]:
        """Build comprehensive normalization mapping.
        
        Returns:
            Dictionary mapping variant names to canonical names
        """
        mapping = {}
        
        # Product Defects variations
        defect_variants = [
            "product defect", "product defects", "defective product",
            "defective products", "broken product", "broken products",
            "damaged product", "damaged products", "faulty product",
            "faulty products", "defect", "defective", "broken", "damaged"
        ]
        for variant in defect_variants:
            mapping[variant.lower()] = self.PRODUCT_DEFECTS
        
        # Billing Errors variations
        billing_variants = [
            "billing error", "billing errors", "billing issue", "billing issues",
            "billing problem", "billing problems", "payment error", "payment errors",
            "payment issue", "payment issues", "payment problem", "payment problems",
            "overcharged", "double charged", "wrong charge", "incorrect charge",
            "charge error", "charge errors", "billing mistake"
        ]
        for variant in billing_variants:
            mapping[variant.lower()] = self.BILLING_ERRORS
        
        # Delivery Delays variations
        delivery_variants = [
            "delivery delay", "delivery delays", "late delivery", "late deliveries",
            "delayed delivery", "delayed deliveries", "shipping delay", "shipping delays",
            "late shipment", "late shipments", "delayed shipment", "delayed shipments",
            "slow delivery", "slow shipping", "delivery problem", "delivery problems",
            "delivery issue", "delivery issues", "late", "delayed"
        ]
        for variant in delivery_variants:
            mapping[variant.lower()] = self.DELIVERY_DELAYS
        
        # Poor Quality variations
        quality_variants = [
            "poor quality", "low quality", "bad quality", "poor product quality",
            "low product quality", "bad product quality", "quality issue",
            "quality issues", "quality problem", "quality problems", "cheap quality",
            "inferior quality", "substandard quality"
        ]
        for variant in quality_variants:
            mapping[variant.lower()] = self.POOR_QUALITY
        
        # Wrong Item variations
        wrong_item_variants = [
            "wrong item", "wrong product", "wrong order", "incorrect item",
            "incorrect product", "incorrect order", "wrong item sent",
            "wrong product sent", "received wrong item", "received wrong product",
            "sent wrong item", "sent wrong product"
        ]
        for variant in wrong_item_variants:
            mapping[variant.lower()] = self.WRONG_ITEM
        
        # Missing Items variations
        missing_variants = [
            "missing item", "missing items", "missing part", "missing parts",
            "missing component", "missing components", "incomplete order",
            "incomplete delivery", "items missing", "parts missing",
            "not included", "item not included"
        ]
        for variant in missing_variants:
            mapping[variant.lower()] = self.MISSING_ITEMS
        
        # Poor Service variations
        service_variants = [
            "poor service", "poor customer service", "bad service",
            "bad customer service", "terrible service", "terrible customer service",
            "rude agent", "rude staff", "unhelpful agent", "unhelpful staff",
            "service quality", "service issue", "service issues",
            "customer service issue", "customer service issues", "poor support"
        ]
        for variant in service_variants:
            mapping[variant.lower()] = self.POOR_SERVICE
        
        # Pricing Issues variations
        pricing_variants = [
            "pricing issue", "pricing issues", "pricing problem", "pricing problems",
            "price issue", "price issues", "price problem", "price problems",
            "expensive", "too expensive", "overpriced", "high price",
            "pricing concern", "pricing concerns", "price concern", "price concerns"
        ]
        for variant in pricing_variants:
            mapping[variant.lower()] = self.PRICING_ISSUES
        
        # Technical Issues variations
        technical_variants = [
            "technical issue", "technical issues", "technical problem",
            "technical problems", "doesn't work", "not working", "won't work",
            "stopped working", "malfunction", "malfunctioning", "tech issue",
            "tech issues", "tech problem", "tech problems"
        ]
        for variant in technical_variants:
            mapping[variant.lower()] = self.TECHNICAL_ISSUES
        
        # Shipping Damage variations
        damage_variants = [
            "shipping damage", "damaged in shipping", "damaged during shipping",
            "damaged in transit", "damaged during delivery", "package damage",
            "damaged package", "broken in shipping", "broken during shipping"
        ]
        for variant in damage_variants:
            mapping[variant.lower()] = self.SHIPPING_DAMAGE
        
        # Return Issues variations
        return_variants = [
            "return issue", "return issues", "return problem", "return problems",
            "refund issue", "refund issues", "refund problem", "refund problems",
            "return request", "return requests", "refund request", "refund requests",
            "return", "refund"
        ]
        for variant in return_variants:
            mapping[variant.lower()] = self.RETURN_ISSUES
        
        # Account Issues variations
        account_variants = [
            "account issue", "account issues", "account problem", "account problems",
            "login issue", "login issues", "login problem", "login problems",
            "access issue", "access issues", "account access", "can't login",
            "cannot login", "account locked"
        ]
        for variant in account_variants:
            mapping[variant.lower()] = self.ACCOUNT_ISSUES
        
        # Generic fallbacks
        generic_variants = [
            "issue", "issues", "problem", "problems", "complaint", "complaints",
            "concern", "concerns", "order issue", "order issues", "order problem",
            "order problems", "item issue", "item issues", "product issue",
            "product issues", "item-related issues", "product-related issues",
            "order", "item", "product", "general issue", "general issues",
            "customer problem", "customer problems", "unknown"
        ]
        for variant in generic_variants:
            mapping[variant.lower()] = self.GENERAL_ISSUE
        
        return mapping
    
    def normalize(self, factor: str) -> str:
        """Normalize a factor name to its canonical category.
        
        Args:
            factor: Factor name to normalize
        
        Returns:
            Canonical factor category name
        """
        if not factor:
            return self.GENERAL_ISSUE
        
        factor_lower = factor.lower().strip()
        
        # Exact match
        if factor_lower in self.normalization_map:
            return self.normalization_map[factor_lower]
        
        # Partial match - check if any mapping key is contained in the factor
        for key, canonical in self.normalization_map.items():
            if key in factor_lower or factor_lower in key:
                return canonical
        
        # If no match found, return as-is (titlecased)
        return factor.strip().title()
    
    def normalize_batch(self, factors: list[str]) -> list[str]:
        """Normalize a list of factors.
        
        Args:
            factors: List of factor names
        
        Returns:
            List of normalized factor names (unique)
        """
        normalized = set()
        for factor in factors:
            if factor:
                normalized.add(self.normalize(factor))
        return sorted(list(normalized))
    
    def get_all_categories(self) -> list[str]:
        """Get all canonical factor categories.
        
        Returns:
            Sorted list of canonical category names
        """
        categories = {
            self.PRODUCT_DEFECTS,
            self.BILLING_ERRORS,
            self.DELIVERY_DELAYS,
            self.POOR_QUALITY,
            self.WRONG_ITEM,
            self.MISSING_ITEMS,
            self.POOR_SERVICE,
            self.PRICING_ISSUES,
            self.TECHNICAL_ISSUES,
            self.SHIPPING_DAMAGE,
            self.RETURN_ISSUES,
            self.ACCOUNT_ISSUES,
        }
        return sorted(list(categories))


# Global normalizer instance
_normalizer = FactorNormalizer()


def normalize_factor(factor: str) -> str:
    """Normalize a single factor using the global normalizer.
    
    Args:
        factor: Factor name to normalize
    
    Returns:
        Canonical factor category name
    """
    return _normalizer.normalize(factor)


def normalize_factors(factors: list[str]) -> list[str]:
    """Normalize a list of factors using the global normalizer.
    
    Args:
        factors: List of factor names
    
    Returns:
        List of normalized factor names (unique)
    """
    return _normalizer.normalize_batch(factors)


if __name__ == "__main__":
    # Demo normalization
    normalizer = FactorNormalizer()
    
    test_factors = [
        "Product Defect",
        "product defects",
        "Broken Product",
        "Damaged Products",
        "Billing Error",
        "billing issues",
        "Payment Problem",
        "Overcharged",
        "Late Delivery",
        "Delivery Delays",
        "Shipping Delay",
        "Poor Quality",
        "Low Quality",
        "Wrong Item",
        "Incorrect Product",
        "Missing Items",
        "Poor Service",
        "Rude Agent",
        "Price Issue",
        "Technical Problem",
        "Won't Work",
        "Unknown",
        "General Issue",
    ]
    
    print("\n" + "=" * 80)
    print("FACTOR NORMALIZATION DEMO")
    print("=" * 80 + "\n")
    
    for factor in test_factors:
        normalized = normalizer.normalize(factor)
        print(f"{factor:<30} → {normalized}")
    
    print("\n" + "=" * 80)
    print("BATCH NORMALIZATION")
    print("=" * 80 + "\n")
    
    print(f"Input: {len(test_factors)} factors")
    normalized_batch = normalizer.normalize_batch(test_factors)
    print(f"Output: {len(normalized_batch)} unique categories\n")
    
    for i, category in enumerate(normalized_batch, 1):
        print(f"  {i}. {category}")
    
    print("\n" + "=" * 80)
    print("ALL CANONICAL CATEGORIES")
    print("=" * 80 + "\n")
    
    for i, category in enumerate(normalizer.get_all_categories(), 1):
        print(f"  {i}. {category}")
