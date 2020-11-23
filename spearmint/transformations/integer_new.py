# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.

import warnings
import numpy as np
from .abstract_transformation import AbstractTransformation

class Integer(AbstractTransformation):

    def __init__(self, num_dims, integer_dimensions, num_values, name="IntegerTransformation"):
        self.name = name
        self.num_dims = num_dims
        self.integer_dimensions = np.array(integer_dimensions)
        self.num_values = np.array(num_values)
        self._validate()
        self.intervals =  np.array([ np.linspace(0,1,value*2+1)[np.linspace(1,value*2-1,value).astype(int)] for value in num_values ])

    def _validate(self):
        assert self.integer_dimensions.shape[0] > 0 and self.num_values.shape[0] > 0, "Integer Transformation error: len of integer_dimensions and num_values must be higher than zero."
        assert self.integer_dimensions.shape[0] == self.num_values.shape[0], "Integer Transformation error: len of integer_dimensions must be the same as the len of num_values."
        assert self.integer_dimensions.shape[0] <= self.num_dims, "Integer transformation error: len of integer_dimensions must be lower or equal that the len of num_dims."

    def truncate(self, inputs):
        self._original_inputs = inputs
        """
        Truncate the inputs to lie between 0 and 1 if it doesn't already.
        This is to make the creation of the intervals to be ok.
        If the inputs genuinely lives outside of [0,1] then we obviously
        don't want to do this, so print out a warning just in case.
        """
        inputs = inputs.copy()
        if np.any(inputs < 0):
            warnings.warn('IntegerTransformation encountered negative values: %s' % inputs[inputs<0])
            inputs[ inputs<0 ] = 0.0
        if np.any(inputs > 1):
            warnings.warn('IntegerTransformation encountered values above 1: %s' % inputs[inputs>1])
            inputs[ inputs>1 ] = 1.0
        
        self.original_normalized_inputs = inputs
 
    @property
    def hypers(self):
        return []

    def return_outputs(self, converted_inputs, original_inputs):
        outputs = np.zeros(original_inputs.shape)
        converted_inputs_index = 0
        for i in range(outputs.shape[ 1 ]):
            if i in self.integer_dimensions:
                outputs[ :,i ] = converted_inputs[ :,converted_inputs_index ]
                converted_inputs_index += 1
            else:
                outputs[ :,i ] = original_inputs[ :,i ]
        return outputs

    def forward_pass_time_timer(self, inputs):
	import time
	start = time.time()
	self.forward_pass_old(inputs)
	end = time.time()
	print("Version old time")
	print(end-start)
	start = time.time()
	self.forward_pass_new(inputs)
	end = time.time()
	print("Version new time")
	print(end-start)
	import pdb; pdb.set_trace();	


    def forward_pass_old(self, inputs):
        self.truncate(inputs)
        inputs_to_convert = inputs[ :, self.integer_dimensions ]
	for k in range(inputs_to_convert.shape[ 0 ]):
		for j in range(self.intervals.shape[ 0 ]):
			for i in range(self.intervals[j].shape[ 0 ]-1):
				if inputs_to_convert[k][j] >= self.intervals[j][i] and inputs_to_convert[k][j] < self.intervals[j][i+1]:
					inputs_to_convert[k][j] = self.intervals[j][i]
					break
        return self.return_outputs(inputs_to_convert, inputs)


    def forward_pass(self, inputs):

        self.truncate(inputs)

        inputs_to_convert = inputs[ :, self.integer_dimensions ]

        inputs[ :, self.integer_dimensions ] = np.array([ np.array([ self.intervals[ j ][ \
            np.argmin(np.absolute(inputs_to_convert[ k ][ j ] - self.intervals[ j ]))] \
            for j in range(inputs_to_convert.shape[ 1 ])]) for k in range(inputs_to_convert.shape[ 0 ])])

        return inputs

    #TODO: Revisar.
    def build_M_matrix(self,V):

        # This avoids problems when doing the evaluations in one data point only

        if len(V.shape) == 1:

            M = np.ones(V.shape[ 0 ])
            M[ self.integer_dimensions ] = 0.0

        else:

            M = np.zeros([V.shape[1],V.shape[2]])

            for i in range(self.num_dims):
                if i not in self.integer_dimensions:
                    M[ :,i ] = 1.0

        return M

    def backward_pass(self, V):
        return V*self.build_M_matrix(V)

